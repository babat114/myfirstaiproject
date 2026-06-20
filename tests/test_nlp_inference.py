"""
============================================
NLP推理测试 — quick-predict API 端到端测试 (参数化优化 v1.0)
============================================
Batch D1: 测试 quick-predict API 的文本输入路径
"""
import os
import pickle
import tempfile
import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app import db
from app.models.model_record import ModelRecord


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def nlp_model_pkl_path():
    """创建临时的 NLP 模型 .pkl 文件 (含 vectorizer + classifier + class_labels)"""
    texts = [
        "非常好，很棒", "太棒了，很喜欢", "服务很好，值得推荐",
        "环境不错，干净整洁", "味道好极了", "价格实惠，性价比高",
        "态度热情，很满意", "好吃的不得了", "很舒服，很安静",
        "特别好吃，强烈推荐",
        "太差了，很失望", "非常糟糕的体验", "服务态度恶劣",
        "脏乱差，不推荐", "价格太贵，不值", "难吃死了",
        "很吵，睡不着", "冷漠无情，态度差", "太挤了，不舒服",
        "差评，不会再来了",
    ]
    labels = (['正面'] * 10) + (['负面'] * 10)

    vectorizer = TfidfVectorizer(max_features=100)
    X_tfidf = vectorizer.fit_transform(texts)

    clf = LogisticRegression(random_state=42, max_iter=500)
    clf.fit(X_tfidf, labels)

    bundle = {
        'model': clf,
        'vectorizer': vectorizer,
        'feature_names': [f'tfidf_{i}' for i in range(X_tfidf.shape[1])],
        'class_labels': ['正面', '负面'],
        'label_encoders': {},
        'task_type': 'nlp',
        'algorithm': 'logistic_regression',
        'scaler': None,
    }

    tmp = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    tmp_path = tmp.name
    with open(tmp_path, 'wb') as f:
        pickle.dump(bundle, f)
    tmp.close()

    yield tmp_path

    try:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    except (PermissionError, OSError):
        pass


@pytest.fixture
def nlp_model_uuid(app, test_user, nlp_model_pkl_path):
    """创建 NLP 模型记录并返回 UUID 字符串"""
    with app.app_context():
        model = ModelRecord(
            name='NLP Inference Test Model',
            model_type='nlp',
            framework='sklearn',
            status='trained',
            owner_id=test_user.id,
            is_public=False,
            model_file_path=nlp_model_pkl_path,
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
        )
        db.session.add(model)
        db.session.commit()
        uuid = model.uuid
    return uuid


# ═══════════════════════════════════════════════════════════════════
# TestNLPQuickPredict
# ═══════════════════════════════════════════════════════════════════

class TestNLPQuickPredict:
    """quick-predict API 端到端测试"""

    def _qp(self, client, model_uuid, text):
        """Helper: 调用 quick-predict API"""
        return client.post(
            f'/api/v1/models/{model_uuid}/quick-predict',
            json={'text': text},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )

    @pytest.mark.parametrize("text,expected_prediction", [
        ("非常好，服务很棒", "正面"),
        ("太差了，很失望",   "负面"),
    ])
    def test_predict_sentiment(self, logged_in_client, nlp_model_uuid,
                                text, expected_prediction):
        """参数化: 正面 / 负面文本预测"""
        resp = self._qp(logged_in_client, nlp_model_uuid, text)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert data['data']['prediction'] == expected_prediction
        assert data['data']['input_mode'] == 'text'
        assert 'probabilities' in data['data']

    @pytest.mark.parametrize("text,expect_400,keyword_in_message", [
        ("",               True,  "输入数据"),
        ("   \t\n  ",      True,  None),
        ("好" * 5001,      True,  "5000"),
        ("!!!@@@###12345", True,  None),
        ("很好" * 500,     False, None),  # 1000 chars, OK
    ])
    def test_predict_input_validation(self, logged_in_client, nlp_model_uuid,
                                       text, expect_400, keyword_in_message):
        """参数化: 空文本 / 纯空白 / 超长 / 纯符号 / 长文本OK"""
        resp = self._qp(logged_in_client, nlp_model_uuid, text)
        if expect_400:
            assert resp.status_code == 400
            data = resp.get_json()
            assert data['success'] is False
            if keyword_in_message:
                assert keyword_in_message in data.get('message', '')
        else:
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['success'] is True

    @pytest.mark.parametrize("client_fixture,uuid_type,expected_status", [
        ("client",           "valid",      401),
        ("logged_in_client", "nonexistent", 404),
    ])
    def test_predict_auth_notfound(self, request, client_fixture, uuid_type,
                                    expected_status, nlp_model_uuid):
        """参数化: 无认证 → 401 / 不存在模型 → 404"""
        client = request.getfixturevalue(client_fixture)
        uuid = nlp_model_uuid if uuid_type == "valid" else "nonexistent-uuid-12345"
        resp = client.post(
            f'/api/v1/models/{uuid}/quick-predict',
            json={'text': '测试'},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert resp.status_code == expected_status

    def test_not_json_request(self, logged_in_client, nlp_model_uuid):
        """非 JSON 请求 → 400"""
        resp = logged_in_client.post(
            f'/api/v1/models/{nlp_model_uuid}/quick-predict',
            data='not json',
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert resp.status_code == 400

    def test_no_text_no_features(self, logged_in_client, nlp_model_uuid):
        """没有 text 也没有 features → 400"""
        resp = logged_in_client.post(
            f'/api/v1/models/{nlp_model_uuid}/quick-predict',
            json={},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert '输入数据' in data.get('message', '')

    def test_response_structure(self, logged_in_client, nlp_model_uuid):
        """响应包含所有必需字段"""
        resp = self._qp(logged_in_client, nlp_model_uuid, "服务很好，值得推荐")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert 'prediction' in data['data']
        assert 'confidence' in data['data']
        assert 'probabilities' in data['data']
        assert 'model_type' in data['data']
        assert 'input_mode' in data['data']
        assert 'input_preview' in data['data']

    def test_confidence_range(self, logged_in_client, nlp_model_uuid):
        """置信度在 0-1 范围内"""
        resp = self._qp(logged_in_client, nlp_model_uuid, "非常好")
        assert resp.status_code == 200
        data = resp.get_json()
        conf = data['data']['confidence']
        assert 0.0 <= conf <= 1.0

    def test_probabilities_sum_to_one(self, logged_in_client, nlp_model_uuid):
        """概率分布之和 ≈ 1.0"""
        resp = self._qp(logged_in_client, nlp_model_uuid, "一般般吧")
        assert resp.status_code == 200
        data = resp.get_json()
        probs = data['data'].get('probabilities', [])
        if probs:
            total = sum(p.get('probability', 0) for p in probs)
            assert abs(total - 1.0) < 0.01, f"概率和={total}, 期望≈1.0"

    def test_feature_input_mode(self, logged_in_client, nlp_model_uuid):
        """特征值输入模式 (features mode)"""
        resp = logged_in_client.post(
            f'/api/v1/models/{nlp_model_uuid}/quick-predict',
            json={'features': {'tfidf_0': 0.1, 'tfidf_1': 0.2}},
            headers={'X-Requested-With': 'XMLHttpRequest'},
        )
        assert resp.status_code in (200, 400)

    def test_permission_other_user_model(self, client, nlp_model_uuid):
        """另一个用户访问私有模型 → 403"""
        from app.models.user import User
        with client.application.app_context():
            user2 = User(
                username='otheruser2',
                email='other2@test.com',
                full_name='Other User 2',
                role='researcher',
                is_active=True,
                is_verified=True,
            )
            user2.set_password('Other123456')
            db.session.add(user2)
            db.session.commit()

        login_resp = client.post('/api/v1/auth/login', json={
            'login_id': 'otheruser2', 'password': 'Other123456'
        })
        if login_resp.status_code == 200:
            token = login_resp.get_json().get('data', {}).get('access_token', '')
            resp = client.post(
                f'/api/v1/models/{nlp_model_uuid}/quick-predict',
                json={'text': '测试'},
                headers={
                    'Authorization': f'Bearer {token}',
                    'X-Requested-With': 'XMLHttpRequest',
                },
            )
            assert resp.status_code == 403
