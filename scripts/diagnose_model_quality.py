"""
多维度模型质量诊断工具

检测维度:
  1. 完美准确率 (accuracy == 1.0) — 可证明过拟合
  2. 近完美准确率 (accuracy >= 0.999) — 高风险
  3. F1-Accuracy 差距 — 类别不平衡过拟合信号
  4. 训练-测试差距 — 训练集过拟合信号 (Phase 1 修复后对新模型生效)
  5. 独立测试集泛化差距 — 外部测试集准确率下降 (Phase 4 重评估后可用)
  6. 低健康评分 — ParameterGuidanceService.analyze_results() health_score < 阈值
  7. 常数预测器 — 对所有输入预测同一类别
  8. 零方差概率 — predict_proba 对所有样本的 std < 0.01

输出: experiments/model_quality_report.json

Usage:
  python scripts/diagnose_model_quality.py                     # 全量诊断
  python scripts/diagnose_model_quality.py --quick             # 仅查最新 100 个模型
  python scripts/diagnose_model_quality.py --dataset Douban    # 按数据集名过滤
  python scripts/diagnose_model_quality.py --no-model-inspect  # 跳过模型文件检查
  python scripts/diagnose_model_quality.py --health-threshold 50  # 自定义健康阈值
"""
import sys
import os
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.dataset import Dataset


# ── 用于常数预测器检测的中文情感测试句 ──
TEST_SENTENCES = [
    "这个产品非常好，我很满意",       # strongly positive
    "太差了，完全不值这个价",          # strongly negative
    "一般般吧，没什么特别的",          # neutral
    "性价比很高，推荐购买",            # positive
    "服务态度恶劣，差评",              # negative
    "物流很快，包装也很好",            # positive
    "跟描述完全不符，上当受骗",        # negative
    "东西还不错，就是有点贵",          # mixed-positive
]

# ── 健康评分阈值 ──
DEFAULT_HEALTH_THRESHOLD = 40  # health_score < 40 视为警告

# ── 准确率阈值 ──
PERFECT_ACC_THRESHOLD = 1.0     # 精确 1.0
NEAR_PERFECT_THRESHOLD = 0.999  # >= 99.9%
F1_ACC_GAP_THRESHOLD = 0.15     # accuracy - f1 > 0.15
TRAIN_TEST_GAP_THRESHOLD = 0.15 # train_acc - test_acc > 0.15
IND_TEST_GAP_SEVERE = 0.30      # 独立测试集准确率下降 >= 30% — 确认过拟合
IND_TEST_GAP_MODERATE = 0.15    # 独立测试集准确率下降 >= 15% — 高风险
ZERO_VAR_PROBA_THRESHOLD = 0.01 # std(proba) < 0.01


def load_model_bundle(path: str) -> Optional[dict]:
    """Load a model bundle (.pkl) and return its contents."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        return bundle if isinstance(bundle, dict) else None
    except Exception:
        return None


def detect_constant_predictor(model_record: ModelRecord) -> Tuple[bool, dict]:
    """Check if model always predicts the same class.

    Returns (is_constant, details_dict).
    """
    path = model_record.model_file_path
    details = {'checked': False, 'error': None}

    bundle = load_model_bundle(path)
    if bundle is None:
        details['error'] = 'model file not found or not loadable'
        return False, details

    clf = bundle.get('model')
    vectorizer = bundle.get('vectorizer')
    class_labels = bundle.get('class_labels', [])

    if clf is None and vectorizer is None:
        details['error'] = 'no model or vectorizer in bundle'
        return False, details

    # Generate test inputs
    try:
        if vectorizer is not None:
            X = vectorizer.transform(TEST_SENTENCES)
            if hasattr(X, 'toarray'):
                X = X.toarray()
        elif clf is not None:
            n_features = getattr(clf, 'n_features_in_', 10)
            rng = np.random.RandomState(42)
            X = rng.randn(len(TEST_SENTENCES), n_features)
        else:
            details['error'] = 'cannot generate test inputs'
            return False, details
    except Exception as e:
        details['error'] = f'test input generation failed: {e}'
        return False, details

    # Get predictions
    try:
        if clf is not None:
            predictions = clf.predict(X)
            unique_preds = np.unique(predictions)
            is_constant = len(unique_preds) == 1

            proba_std = None
            zero_var_proba = False
            if hasattr(clf, 'predict_proba'):
                try:
                    proba = clf.predict_proba(X)
                    proba_std = float(np.std(proba[:, 0])) if proba.shape[1] >= 2 else 0.0
                    zero_var_proba = proba_std < ZERO_VAR_PROBA_THRESHOLD
                except Exception:
                    pass

            details = {
                'checked': True,
                'n_test_inputs': len(TEST_SENTENCES),
                'unique_predictions': int(len(unique_preds)),
                'predicted_class': str(unique_preds[0]) if is_constant else None,
                'class_labels': [str(c) for c in class_labels] if class_labels else [],
                'proba_std': proba_std,
                'zero_variance_proba': zero_var_proba,
            }
            return (is_constant or zero_var_proba), details
        else:
            details['error'] = 'no classifier in bundle (vectorizer only)'
            return False, details
    except Exception as e:
        details['error'] = f'prediction failed: {e}'
        return False, details


def compute_health_score(model_record: ModelRecord) -> Optional[dict]:
    """Compute health score using ParameterGuidanceService."""
    try:
        from app.services.parameter_guidance_service import ParameterGuidanceService

        metrics_dict = model_record.metrics_dict
        hyperparams = model_record.hyperparameters_dict

        # Build metrics_history from saved epoch data if available
        metrics_history = metrics_dict.get('history', [])
        final_metrics = {
            k: v for k, v in metrics_dict.items()
            if k not in ('history', 'feature_names', 'label_encoders')
        }

        dataset_info = {}
        if model_record.training_dataset:
            ds = model_record.training_dataset
            dataset_info = {
                'name': ds.name,
                'row_count': getattr(ds, 'row_count', None),
                'column_count': getattr(ds, 'column_count', None),
                'category': getattr(ds, 'category', None),
            }

        result = ParameterGuidanceService.analyze_results(
            metrics_history=metrics_history,
            final_metrics=final_metrics,
            dataset_info=dataset_info,
            hyperparams=hyperparams,
        )
        return result
    except Exception as e:
        return {'error': str(e), 'health_score': None}


def diagnose_model(model_record: ModelRecord, inspect_model: bool = True,
                   health_threshold: int = DEFAULT_HEALTH_THRESHOLD) -> dict:
    """Run all diagnostic checks on a single model.

    Returns a dict with reasons list and severity tier.
    """
    reasons = []
    warnings = []
    details = {}

    acc = model_record.accuracy
    f1 = model_record.f1_score
    metrics_dict = model_record.metrics_dict

    # 1. Perfect accuracy check
    if acc is not None and acc == PERFECT_ACC_THRESHOLD:
        reasons.append('perfect_accuracy')
        details['accuracy'] = acc

    # 2. Near-perfect accuracy check
    if acc is not None and PERFECT_ACC_THRESHOLD > acc >= NEAR_PERFECT_THRESHOLD:
        reasons.append('near_perfect_accuracy')
        details['accuracy'] = acc

    # 3. F1-Accuracy gap check
    if acc is not None and f1 is not None and (acc - f1) > F1_ACC_GAP_THRESHOLD:
        reasons.append('f1_accuracy_gap')
        details['f1_accuracy_gap'] = round(acc - f1, 4)

    # 3b. NULL f1 + high accuracy → majority-class predictor (严重不平衡)
    if acc is not None and f1 is None and acc >= 0.90:
        reasons.append('missing_f1')
        details['missing_f1'] = 'f1_score is NULL with high accuracy — likely majority-class predictor'
        # 尝试从 metrics_json 中找 f1
        if 'f1_score' in metrics_dict:
            details['f1_from_metrics'] = metrics_dict['f1_score']


    # 3c. Explicit majority-class predictor: high acc but near-zero f1
    if acc is not None and f1 is not None and acc >= 0.90 and f1 < 0.01:
        reasons.append('majority_class_predictor')
        details['f1_accuracy_gap'] = round(acc - f1, 4)

    # 4. Train-test gap check
    train_acc = metrics_dict.get('train_accuracy')
    test_acc = metrics_dict.get('test_accuracy', acc)
    if train_acc is not None and test_acc is not None:
        gap = train_acc - test_acc
        details['train_test_gap'] = round(gap, 4)
        if gap > TRAIN_TEST_GAP_THRESHOLD:
            reasons.append('train_test_gap')

    # 5. Independent test gap — 独立测试集泛化差距 (weighted by collection_method)
    ind_acc = model_record.independent_accuracy
    collection_method = None
    if model_record.independent_test_dataset_id:
        from app.models.dataset import Dataset as DsModel
        ind_ds = DsModel.query.get(model_record.independent_test_dataset_id)
        if ind_ds:
            collection_method = ind_ds.collection_method
            details['independent_collection_method'] = collection_method

    if acc is not None and ind_acc is not None:
        gap = acc - ind_acc
        details['independent_test_gap'] = round(gap, 4)
        # 根据采集方式调整严重性: URL来源的数据权重更高
        # synthetic 来源的阈值放宽 1.5x (因为合成数据本身来自训练分布)
        if collection_method == 'synthetic':
            severe_threshold = IND_TEST_GAP_SEVERE * 1.5  # 45% for synthetic
            moderate_threshold = IND_TEST_GAP_MODERATE * 1.5  # 22.5% for synthetic
        else:
            severe_threshold = IND_TEST_GAP_SEVERE   # 30%
            moderate_threshold = IND_TEST_GAP_MODERATE  # 15%

        if gap >= severe_threshold:
            reasons.append('independent_test_gap_severe')
        elif gap >= moderate_threshold:
            reasons.append('independent_test_gap_moderate')
    elif ind_acc is None:
        # Only warn about missing independent evaluation if model has high accuracy
        # (suspicious of overfitting) — not a warning for normal-performing models
        if acc is not None and acc >= 0.90:
            warnings.append('no_independent_evaluation')
            details['independent_test_status'] = 'not_evaluated_high_acc'
        else:
            details['independent_test_status'] = 'not_evaluated'
    else:
        details['independent_test_status'] = 'evaluated'
        details['independent_test_gap'] = round(acc - ind_acc, 4) if acc and ind_acc else None

    # 6. Health score
    health_result = compute_health_score(model_record)
    health_score = health_result.get('health_score') if health_result else None
    details['health_score'] = health_score
    if health_score is not None and health_score < health_threshold:
        warnings.append(f'low_health_score({health_score})')

    # 7. Constant predictor + zero variance proba (requires model file)
    if inspect_model and model_record.model_file_path:
        is_broken, cp_details = detect_constant_predictor(model_record)
        details['constant_predictor'] = cp_details
        if is_broken:
            reasons.append('constant_or_zero_var')
    else:
        details['constant_predictor'] = {'checked': False}

    # Determine tier
    # Tier 1 (definite_delete): constant predictor, perfect accuracy, or severe independent test gap
    # Tier 2 (high_risk): near-perfect, F1 gap, train-test gap, or moderate independent test gap
    # Tier 3 (warning): low health score but accuracy < 0.95
    definite_reasons = {'constant_or_zero_var', 'perfect_accuracy', 'independent_test_gap_severe', 'missing_f1', 'majority_class_predictor'}
    high_risk_reasons = {'near_perfect_accuracy', 'f1_accuracy_gap', 'train_test_gap',
                         'independent_test_gap_moderate'}

    has_definite = any(r in definite_reasons for r in reasons)
    has_high_risk = any(r in high_risk_reasons for r in reasons)
    has_warning = len(warnings) > 0

    if has_definite:
        tier = 'definite_delete'
    elif has_high_risk and acc is not None and acc >= 0.95:
        tier = 'high_risk_delete'
    elif has_warning and (acc is None or acc < 0.95):
        tier = 'warning_only'
    elif has_high_risk:
        tier = 'warning_only'
    else:
        tier = 'healthy'

    return {
        'model_uuid': model_record.uuid,
        'model_id': model_record.id,
        'model_name': model_record.name,
        'model_type': model_record.model_type,
        'framework': model_record.framework,
        'algorithm': model_record.hyperparameters_dict.get('algorithm', ''),
        'status': model_record.status,
        'accuracy': acc,
        'precision': model_record.precision,
        'recall': model_record.recall,
        'f1_score': f1,
        'dataset_name': model_record.training_dataset.name if model_record.training_dataset else None,
        'tier': tier,
        'reasons': reasons,
        'warnings': warnings,
        'details': details,
    }


def main():
    parser = argparse.ArgumentParser(description='多维度模型质量诊断')
    parser.add_argument('--output-json', default=None,
                        help='输出 JSON 报告路径 (默认: experiments/model_quality_report.json)')
    parser.add_argument('--quick', action='store_true',
                        help='仅检查最新 100 个已训练模型')
    parser.add_argument('--dataset', type=str, default=None,
                        help='按数据集名称过滤 (模糊匹配)')
    parser.add_argument('--no-model-inspect', action='store_true',
                        help='跳过模型文件加载 (快速模式，但漏检常数预测器)')
    parser.add_argument('--health-threshold', type=int, default=DEFAULT_HEALTH_THRESHOLD,
                        help=f'健康评分警告阈值 (默认: {DEFAULT_HEALTH_THRESHOLD})')
    parser.add_argument('--model-type', type=str, default=None,
                        help='按模型类型过滤 (classification/regression/clustering/nlp/...)')
    parser.add_argument('--status', type=str, default='trained',
                        help='按状态过滤 (默认: trained)')
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        output_path = args.output_json
        if output_path is None:
            experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'
            experiments_dir.mkdir(exist_ok=True)
            output_path = str(experiments_dir / 'model_quality_report.json')

        inspect_model = not args.no_model_inspect
        health_threshold = args.health_threshold

        # ── Build query ──
        query = ModelRecord.query
        if args.status:
            query = query.filter_by(status=args.status)
        if args.model_type:
            query = query.filter_by(model_type=args.model_type)

        all_models = query.order_by(ModelRecord.created_at.desc()).all()

        # Filter by dataset name if specified
        if args.dataset:
            subset = []
            filter_lower = args.dataset.lower()
            for m in all_models:
                ds_name = (m.training_dataset.name if m.training_dataset else '').lower()
                if filter_lower in ds_name:
                    subset.append(m)
            models = subset
        else:
            models = list(all_models)

        if args.quick and len(models) > 100:
            models = models[:100]

        total = len(models)
        print(f"{'=' * 70}")
        print(f"  模型质量诊断工具 v1.0")
        print(f"  待诊断模型: {total} 个")
        print(f"  模型文件检查: {'开启' if inspect_model else '关闭'}")
        print(f"  健康评分阈值: {health_threshold}")
        if args.dataset:
            print(f"  数据集过滤: {args.dataset}")
        if args.model_type:
            print(f"  模型类型过滤: {args.model_type}")
        print(f"{'=' * 70}")

        # ── Diagnose each model ──
        results = []
        definite_delete = []
        high_risk_delete = []
        warning_only = []
        healthy = []
        errors = []

        for i, m in enumerate(models):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  进度: {i + 1}/{total}...")

            try:
                result = diagnose_model(m, inspect_model=inspect_model,
                                       health_threshold=health_threshold)
                results.append(result)

                tier = result['tier']
                if tier == 'definite_delete':
                    definite_delete.append(result)
                elif tier == 'high_risk_delete':
                    high_risk_delete.append(result)
                elif tier == 'warning_only':
                    warning_only.append(result)
                else:
                    healthy.append(result)
            except Exception as e:
                errors.append({
                    'model_id': m.id,
                    'model_uuid': m.uuid,
                    'model_name': m.name,
                    'error': str(e),
                })

        # ── Group by dataset ──
        by_dataset = {}
        for r in definite_delete + high_risk_delete:
            ds = r['dataset_name'] or 'unknown'
            by_dataset.setdefault(ds, {'definite': 0, 'high_risk': 0, 'models': []})
            if r['tier'] == 'definite_delete':
                by_dataset[ds]['definite'] += 1
            else:
                by_dataset[ds]['high_risk'] += 1
            by_dataset[ds]['models'].append(r['model_name'])

        # ── Build report ──
        report = {
            'generated_at': datetime.now().isoformat(),
            'config': {
                'health_threshold': health_threshold,
                'model_inspect': inspect_model,
                'perfect_acc_threshold': PERFECT_ACC_THRESHOLD,
                'near_perfect_threshold': NEAR_PERFECT_THRESHOLD,
                'ind_test_gap_severe': IND_TEST_GAP_SEVERE,
                'ind_test_gap_moderate': IND_TEST_GAP_MODERATE,
            },
            'summary': {
                'total_models_checked': total,
                'definite_delete': len(definite_delete),
                'high_risk_delete': len(high_risk_delete),
                'warning_only': len(warning_only),
                'healthy': len(healthy),
                'errors': len(errors),
            },
            'definite_delete': definite_delete,
            'high_risk_delete': high_risk_delete,
            'warning_only': warning_only,
            'by_dataset': {
                ds: {
                    'definite': v['definite'],
                    'high_risk': v['high_risk'],
                    'models': v['models'],
                }
                for ds, v in sorted(by_dataset.items(), key=lambda x: -(x[1]['definite'] + x[1]['high_risk']))
            },
            'errors': errors,
        }

        # ── Write report ──
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # ── Print summary ──
        print(f"\n{'=' * 70}")
        print(f"  诊断完成")
        print(f"{'=' * 70}")
        print(f"  必定删除 (Tier 1):  {len(definite_delete):>4} 个  (常数预测器 / 准确率=1.0)")
        print(f"  高风险   (Tier 2):  {len(high_risk_delete):>4} 个  (准确率>=99.9% + 其他信号)")
        print(f"  仅警告   (Tier 3):  {len(warning_only):>4} 个  (低健康评分)")
        print(f"  健康:               {len(healthy):>4} 个")
        if errors:
            print(f"  错误:               {len(errors):>4} 个")
        print(f"{'=' * 70}")

        if definite_delete:
            print(f"\n[!!] Tier 1 -- 必定删除 ({len(definite_delete)} 个):")
            for r in definite_delete[:20]:
                reasons_str = ', '.join(r['reasons'])
                print(f"  [{r['model_name'][:40]:40s}] acc={str(r['accuracy']):6s}  "
                      f"dataset={r['dataset_name'] or 'N/A':25s}  reasons=[{reasons_str}]")
            if len(definite_delete) > 20:
                print(f"  ... 还有 {len(definite_delete) - 20} 个")

        if high_risk_delete:
            print(f"\n[!] Tier 2 -- 高风险 ({len(high_risk_delete)} 个):")
            for r in high_risk_delete[:10]:
                reasons_str = ', '.join(r['reasons'])
                print(f"  [{r['model_name'][:40]:40s}] acc={str(r['accuracy']):6s}  "
                      f"dataset={r['dataset_name'] or 'N/A':25s}  reasons=[{reasons_str}]")
            if len(high_risk_delete) > 10:
                print(f"  ... 还有 {len(high_risk_delete) - 10} 个")

        if warning_only:
            print(f"\n[*] Tier 3 -- 仅警告 ({len(warning_only)} 个):")
            for r in warning_only[:5]:
                hs = r['details'].get('health_score')
                print(f"  [{r['model_name'][:40]:40s}] health_score={hs}  "
                      f"dataset={r['dataset_name'] or 'N/A'}")
            if len(warning_only) > 5:
                print(f"  ... 还有 {len(warning_only) - 5} 个")

        print(f"\n[Report] 完整报告已保存: {output_path}")

        if definite_delete or high_risk_delete:
            total_bad = len(definite_delete) + len(high_risk_delete)
            print(f"\n[Next] 下一步: python scripts/clean_overfit_models.py --report-json {output_path} --dry-run")


if __name__ == '__main__':
    main()
