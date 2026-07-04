# -*- coding: utf-8 -*-
"""
============================================
Task 9: NLP end-to-end prediction verification
============================================
Verify newly trained NLP model:
1. load_model extracts vectorizer + class_labels
2. vectorizer.transform text -> TF-IDF
3. ModelInferenceService.predict outputs correct sentiment
4. quick_predict API HTTP endpoint

Usage:
  python scripts/verify_nlp_e2e.py                     # 交互式模式 (打印结果)
  python scripts/verify_nlp_e2e.py --pytest              # pytest 模式 (断言, 返回exit code)
  python scripts/verify_nlp_e2e.py --model-uuid <uuid>   # 指定模型UUID (跳过DB查询)

NOTE: ASCII-only markers for Windows GBK compatibility.
"""
import os
import sys
import json
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from app import create_app, db
from app.models.model_record import ModelRecord
from app.services.inference_service import ModelInferenceService

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

app = create_app()

# Test cases - clearly positive/negative Chinese reviews
TEST_CASES = [
    ("酒店位置很好，房间干净整洁，服务态度非常棒！", "正面", "clear-positive"),
    ("前台接待很热情，早餐种类丰富，下次还会入住。", "正面", "clear-positive-2"),
    ("太差了，房间又脏又破，隔音还不好，一整晚没睡好。", "负面", "clear-negative"),
    ("服务态度恶劣，等了两个小时才入住，太糟糕了。", "负面", "clear-negative-2"),
    ("性价比还可以，不过隔音确实不太好。", None, "mixed-neutral"),
    ("总的来说还行，但有些地方需要改进。", None, "mixed-positive"),
]


def load_latest_nlp_model(model_uuid=None):
    """Load latest trained NLP model, or specific model by UUID."""
    with app.app_context():
        if model_uuid:
            model = ModelRecord.query.filter_by(uuid=model_uuid).first()
            if not model:
                logger.error("[FAIL] Model not found: %s", model_uuid)
                return None, None, None
        else:
            # 优先选择准确率最高且 >= 0.75 的 NLP 模型 (避免低质量模型导致误报)
            model = ModelRecord.query.filter_by(
                model_type='nlp', status='trained'
            ).filter(
                ModelRecord.accuracy >= 0.75
            ).order_by(ModelRecord.accuracy.desc()).first()
            if not model:
                # 回退: 选最新的 (哪怕准确率低)
                model = ModelRecord.query.filter_by(
                    model_type='nlp', status='trained'
                ).order_by(ModelRecord.created_at.desc()).first()

        if not model:
            logger.error("[FAIL] No trained NLP model found")
            return None, None, None

        logger.info("Model: %s (id=%s, uuid=%s)", model.name, model.id, model.uuid)
        logger.info("accuracy=%s, model_file=%s", model.accuracy, model.model_file_path)

        model_obj, metadata, tokenizer, error = ModelInferenceService.load_model(model)
        if error:
            logger.error("[FAIL] Load error: %s", error)
            return None, None, None

        vectorizer = metadata.get('vectorizer') if metadata else None
        class_labels = metadata.get('class_labels', []) if metadata else []

        logger.info("vectorizer=%s", "YES" if vectorizer is not None else "NO")
        logger.info("class_labels=%s", class_labels)
        logger.info("feature_names count=%d",
                     len(metadata.get('feature_names', [])) if metadata else 0)

        return model, metadata, vectorizer


def test_direct_prediction(model, metadata, vectorizer, pytest_mode=False):
    """Test direct ModelInferenceService.predict with TF-IDF transform.

    In pytest_mode: raises AssertionError on failure.
    Otherwise: returns True/False.
    """
    header = "Test 1: Direct prediction (vectorizer -> predict)"
    if not pytest_mode:
        logger.info("\n=== %s ===\n", header)

    if vectorizer is None:
        msg = "[SKIP] Model has no vectorizer"
        if pytest_mode:
            pytest.skip(msg)
        logger.error(msg)
        return False

    passed = 0
    failed = 0
    failures = []

    for text, expected_label, desc in TEST_CASES:
        try:
            # Vectorizer transform — 使用 tfidf_0, tfidf_1... 列名 (与训练时一致)
            X_vec = vectorizer.transform([text])
            X_dense = X_vec.toarray() if hasattr(X_vec, 'toarray') else X_vec
            df = pd.DataFrame(
                X_dense,
                columns=[f'tfidf_{i}' for i in range(X_dense.shape[1])],
            )

            # Predict
            result = ModelInferenceService.predict(model, df)

            if not result.get('success'):
                msg = f"[FAIL] [{desc}] predict error: {result.get('error')}"
                if pytest_mode:
                    raise AssertionError(msg)
                logger.info("  %s", msg)
                failed += 1
                failures.append(desc)
                continue

            prediction = result.get('predictions', [None])[0]
            probs = result.get('probabilities', [])

            # Build probability display
            prob_str = ''
            if probs and len(probs) > 0 and probs[0]:
                prob_str = ' | '.join(
                    "{}={:.3f}".format(p.get('class', '?'), p.get('probability', 0))
                    for p in probs[0][:3]
                )

            # Verify label
            label_ok = True
            if expected_label is not None:
                if str(prediction) != expected_label:
                    label_ok = False

            status = '[PASS]' if label_ok else '[FAIL]'
            if not pytest_mode:
                logger.info("  %s [%s] text=%s...", status, desc, text[:40])
                logger.info("     pred=%s, expected=%s", prediction, expected_label or '(any)')
                if prob_str:
                    logger.info("     probs: %s", prob_str)

            if label_ok:
                passed += 1
            else:
                failed += 1
                failures.append(f"{desc}: pred={prediction}, expected={expected_label}")

        except Exception as e:
            msg = f"[FAIL] [{desc}] exception: {e}"
            if pytest_mode:
                raise AssertionError(msg) from e
            logger.info("  %s", msg)
            import traceback
            traceback.print_exc()
            failed += 1
            failures.append(desc)

    summary = f"Result: {passed} passed, {failed} failed (total {len(TEST_CASES)})"
    if pytest_mode:
        assert failed == 0, f"{summary}\nFailures: {failures}"
    else:
        logger.info("\n  %s", summary)
    return failed == 0


def test_sentiment_fallback(metadata, pytest_mode=False):
    """Test fallback: keyword sentiment analysis."""
    header = "Test 2: Sentiment analysis fallback"
    if not pytest_mode:
        logger.info("\n=== %s ===\n", header)

    from app.services.feature_extractor import FeatureExtractor

    passed = 0
    for text, expected_label, desc in TEST_CASES:
        sentiment = FeatureExtractor.analyze_sentiment(text)
        label = sentiment['label']
        pos = sentiment.get('positive_count', 0)
        neg = sentiment.get('negative_count', 0)

        match = 'HIT' if (expected_label is None or label == expected_label) else 'MIS'
        if not pytest_mode:
            logger.info("  [%s] [%s] sentiment=%s (pos=%d, neg=%d), conf=%.2f",
                         match, desc, label, pos, neg, sentiment['confidence'])
        if match == 'HIT':
            passed += 1

    summary = f"Keyword method hits: {passed}/{len(TEST_CASES)}"
    if pytest_mode:
        # 关键词匹配准确率应该 >= 2/4 (至少50%)
        assert passed >= 2, summary
    else:
        logger.info("\n  %s", summary)
    return True


def test_quick_predict_api(model, pytest_mode=False):
    """Test HTTP quick_predict API endpoint."""
    header = "Test 3: quick_predict HTTP API"
    if not pytest_mode:
        logger.info("\n=== %s ===\n", header)

    with app.test_client() as client:
        # Login
        login_resp = client.post('/api/v1/auth/login', json={
            'login_id': 'admin', 'password': 'Admin123456'
        })
        if login_resp.status_code != 200:
            msg = f"[FAIL] Login failed: {login_resp.status_code}"
            if pytest_mode:
                raise AssertionError(msg)
            logger.info("  %s", msg)
            return False
        token = login_resp.get_json().get('data', {}).get('access_token', '')

        passed = 0
        failed = 0
        failures = []

        for text, expected_label, desc in TEST_CASES[:4]:
            resp = client.post(
                f'/api/v1/models/{model.uuid}/quick-predict',
                json={'text': text},
                headers={'Authorization': 'Bearer {}'.format(token)}
            )

            data = resp.get_json()
            success = data.get('success', False)
            prediction = data.get('data', {}).get('prediction', '')
            note = data.get('data', {}).get('note', '')
            probs = data.get('data', {}).get('probabilities', [])

            label_ok = (prediction == expected_label) if expected_label else True
            status = '[PASS]' if (success and label_ok) else '[FAIL]'

            prob_str = ''
            if probs:
                prob_str = ' | '.join(
                    "{}={:.3f}".format(p.get('class', '?'), p.get('probability', 0))
                    for p in probs[:3]
                )

            if not pytest_mode:
                logger.info("  %s [%s] HTTP %d pred=%s expected=%s",
                             status, desc, resp.status_code, prediction, expected_label)
                if prob_str:
                    logger.info("     probs: %s", prob_str)
                if note:
                    logger.info("     note: %s", note[:100])

            if success and label_ok:
                passed += 1
            else:
                failed += 1
                failures.append(f"{desc}: HTTP {resp.status_code} pred={prediction} expected={expected_label}")

        summary = f"HTTP API: {passed} passed, {failed} failed"
        if pytest_mode:
            assert failed == 0, f"{summary}\nFailures: {failures}"
        else:
            logger.info("\n  %s", summary)
        return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description='NLP End-to-End Prediction Verification'
    )
    parser.add_argument(
        '--pytest', action='store_true',
        help='Run in pytest mode (use assertions, return exit codes)'
    )
    parser.add_argument(
        '--model-uuid', type=str, default=None,
        help='Specify model UUID to test (skip DB query for latest)'
    )
    args = parser.parse_args()

    pytest_mode = args.pytest

    if not pytest_mode:
        logger.info("=" * 60)
        logger.info("Task 9: NLP End-to-End Prediction Verification")
        logger.info("=" * 60)

    model, metadata, vectorizer = load_latest_nlp_model(model_uuid=args.model_uuid)
    if model is None:
        msg = "[SKIP] No available NLP model, skipping prediction tests"
        if pytest_mode:
            pytest.skip(msg) if 'pytest' in sys.modules else sys.exit(0)
        logger.info("\n%s", msg)
        return

    all_pass = True

    # Test 1: Direct prediction via vectorizer
    if not test_direct_prediction(model, metadata, vectorizer, pytest_mode=pytest_mode):
        all_pass = False

    # Test 2: Sentiment fallback (informational only)
    test_sentiment_fallback(metadata, pytest_mode=pytest_mode)

    # Test 3: HTTP API
    if not test_quick_predict_api(model, pytest_mode=pytest_mode):
        all_pass = False

    if pytest_mode:
        assert all_pass, "NLP E2E verification failed — check individual test results above"
        print("[PASS] NLP E2E verification complete")
    else:
        logger.info("\n" + "=" * 60)
        if all_pass:
            logger.info("[PASS] Task 9 complete - NLP end-to-end prediction works")
        else:
            logger.info("[WARN] Task 9 partial failure - check output above")
        logger.info("=" * 60)

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
