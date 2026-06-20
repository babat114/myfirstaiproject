"""
============================================
评论模块测试 (参数化优化 v1.0)
测试评论的 CRUD、审核、权限
============================================
"""
import pytest
import time
from app import db
from app.models.user import User
from app.models.model_record import ModelRecord
from app.models.comment import Comment
from app.services.comment_service import CommentService
from app.utils.jwt_helpers import generate_access_token


@pytest.fixture
def public_model(app, test_admin):
    """创建公开模型"""
    model = ModelRecord(
        name='Test Public Model',
        model_type='classification',
        is_public=True,
        owner_id=test_admin.id,
        status='trained',
    )
    db.session.add(model)
    db.session.commit()
    return model


@pytest.fixture
def private_model(app, test_admin):
    """创建私有模型"""
    model = ModelRecord(
        name='Test Private Model',
        model_type='regression',
        is_public=False,
        owner_id=test_admin.id,
        status='draft',
    )
    db.session.add(model)
    db.session.commit()
    return model


@pytest.fixture
def other_user(app):
    """创建另一个普通用户"""
    user = User(
        username='other_test', email='other@test.com',
        role='researcher', is_active=True,
    )
    user.set_password('Other123456')
    db.session.add(user)
    db.session.commit()
    return user


class TestCommentService:
    """评论服务测试"""

    def test_add_comment_success(self, app, test_user, public_model):
        """测试成功发表评论"""
        comment, error = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='这是一个很好的模型！',
        )
        assert comment is not None
        assert error is None
        assert comment.is_visible is True
        assert comment.content == '这是一个很好的模型！'

    @pytest.mark.parametrize("content", [
        "你这个傻逼模型",
        "加微信 abc123 了解更多",
        "This is fucking garbage shit!",
        "fuckyou",
    ])
    def test_add_comment_filtered(self, app, test_user, public_model, content):
        """参数化: 中文辱骂 / 广告 / 英文辱骂 / 拼接辱骂词"""
        comment, error = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content=content,
        )
        assert comment is not None
        assert comment.is_visible is False

    def test_private_model_no_comment(self, app, test_user, private_model):
        """测试私有模型不可评论"""
        comment, error = CommentService.add_comment(
            user=test_user, model_id=private_model.id,
            content='正常评论',
        )
        assert comment is None
        assert error is not None
        assert '公开' in error

    def test_get_comments_for_model(self, app, test_user, public_model):
        """测试获取模型评论列表"""
        for i in range(3):
            CommentService.add_comment(
                user=test_user, model_id=public_model.id,
                content=f'评论 {i+1}',
            )
        result = CommentService.get_comments_for_model(
            model_id=public_model.id, user=test_user,
        )
        assert result['total'] == 3
        assert len(result['comments']) == 3

    def test_comments_sorted_by_newest(self, app, test_user, public_model):
        """测试评论按最新优先排序"""
        CommentService.add_comment(user=test_user, model_id=public_model.id, content='旧评论')
        time.sleep(1.0)  # SQLite 秒级精度, 确保时间戳不同
        CommentService.add_comment(user=test_user, model_id=public_model.id, content='新评论')

        result = CommentService.get_comments_for_model(
            model_id=public_model.id, user=test_user,
        )
        assert result['comments'][0]['content'] == '新评论'

    @pytest.mark.parametrize("actor_fixture,expect_success,permanent,expected_in_db", [
        ("test_user",  True,  False, "soft_deleted"),
        ("other_user", False, False, "unchanged"),
        ("test_admin", True,  True,  "physical_deleted"),
    ])
    def test_delete_comment_by_role(self, app, request, test_user, other_user,
                                     test_admin, public_model, actor_fixture,
                                     expect_success, permanent, expected_in_db):
        """参数化: 自己删除 / 他人删除 / 管理员永久删除"""
        actor = request.getfixturevalue(actor_fixture)
        comment, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='待删除评论',
        )
        cid = comment.id

        success, error = CommentService.delete_comment(
            comment_id=cid, user=actor, permanent=permanent,
        )
        assert success == expect_success

        c = db.session.get(Comment, cid)
        if expected_in_db == "soft_deleted":
            assert c.is_visible is False
            assert c.moderation_reason == 'owner_deleted'
        elif expected_in_db == "physical_deleted":
            assert c is None
        elif expected_in_db == "unchanged":
            assert c.is_visible is True

    @pytest.mark.parametrize("actor_fixture,expect_success", [
        ("test_admin", True),
        ("test_user",  False),
    ])
    def test_restore_comment(self, app, request, test_user, test_admin,
                              public_model, actor_fixture, expect_success):
        """参数化: 管理员恢复 / 非管理员不能恢复"""
        actor = request.getfixturevalue(actor_fixture)
        comment, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='待恢复评论',
        )
        # 软删除
        CommentService.delete_comment(comment_id=comment.id, user=test_user)

        success, error = CommentService.restore_comment(
            comment_id=comment.id, user=actor,
        )
        assert success == expect_success
        if not expect_success:
            assert '管理员' in error

    def test_html_stripped_from_comment(self, app, test_user, public_model):
        """测试HTML标签被移除"""
        comment, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='<script>alert("xss")</script>正常内容',
        )
        assert '<script>' not in comment.content
        assert '正常内容' in comment.content

    def test_empty_comment_rejected(self, app, test_user, public_model):
        """测试空评论被拒绝"""
        comment, error = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='   ',
        )
        assert comment is None
        assert error is not None

    def test_reply_to_comment(self, app, test_user, other_user, public_model):
        """测试回复评论"""
        parent, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='主评论',
        )
        reply, error = CommentService.add_comment(
            user=other_user, model_id=public_model.id,
            content='回复评论', parent_id=parent.id,
        )
        assert reply is not None
        assert reply.parent_id == parent.id
        assert reply.is_reply is True
        assert parent.reply_count == 1

    def test_comment_max_length(self, app, test_user, public_model):
        """测试评论长度限制 (2000字)"""
        long_content = 'A' * 3000
        comment, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content=long_content,
        )
        assert len(comment.content) <= 2000

    def test_pagination(self, app, test_user, public_model):
        """测试评论分页"""
        for i in range(25):
            CommentService.add_comment(
                user=test_user, model_id=public_model.id,
                content=f'评论 {i+1}',
            )

        result = CommentService.get_comments_for_model(
            model_id=public_model.id, user=test_user,
            page=1, per_page=10,
        )
        assert len(result['comments']) == 10
        assert result['total'] == 25
        assert result['pages'] == 3
        assert result['has_next'] is True

        result = CommentService.get_comments_for_model(
            model_id=public_model.id, user=test_user,
            page=3, per_page=10,
        )
        assert len(result['comments']) == 5
        assert result['has_next'] is False


class TestCommentAPI:
    """评论 API 测试"""

    @staticmethod
    def _token(user) -> str:
        """生成JWT token"""
        return generate_access_token(user.id, user.username, user.role)

    def test_list_comments_api(self, app, client, test_user, public_model):
        """测试API获取评论列表"""
        token = self._token(test_user)

        resp = client.get(
            f'/api/v1/models/{public_model.id}/comments',
            headers={'Authorization': f'Bearer {token}'},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True

    @pytest.mark.parametrize("content,expect_flagged", [
        ("API测试评论",   False),
        ("垃圾傻逼模型",  True),
    ])
    def test_add_comment_api(self, app, client, test_user, public_model,
                              content, expect_flagged):
        """参数化: 正常评论 / 违规评论被标记"""
        token = self._token(test_user)

        resp = client.post(
            f'/api/v1/models/{public_model.id}/comments',
            json={'content': content},
            headers={'Authorization': f'Bearer {token}'},
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data['success'] is True
        # flagged may be absent for normal comments
        assert data.get('flagged', False) == expect_flagged

    def test_delete_comment_api(self, app, client, test_user, public_model):
        """测试API删除评论"""
        token = self._token(test_user)

        comment, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='待删除的API测试评论',
        )

        resp = client.delete(
            f'/api/v1/comments/{comment.id}',
            headers={'Authorization': f'Bearer {token}'},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True

    def test_get_replies_api(self, app, client, test_user, other_user, public_model):
        """测试API获取回复"""
        token = self._token(test_user)

        parent, _ = CommentService.add_comment(
            user=test_user, model_id=public_model.id,
            content='主评论',
        )
        CommentService.add_comment(
            user=other_user, model_id=public_model.id,
            content='回复1', parent_id=parent.id,
        )

        resp = client.get(
            f'/api/v1/comments/{parent.id}/replies',
            headers={'Authorization': f'Bearer {token}'},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data['data']) == 1


class TestCommentModel:
    """Comment 模型测试"""

    def test_comment_to_dict(self, app, test_user, public_model):
        """测试序列化"""
        comment = Comment(
            model_id=public_model.id,
            user_id=test_user.id,
            content='测试序列化',
        )
        db.session.add(comment)
        db.session.commit()

        d = comment.to_dict()
        assert d['content'] == '测试序列化'
        assert d['is_visible'] is True
        assert d['is_reply'] is False
        assert d['reply_count'] == 0
        assert d['user']['username'] == 'testuser'

    @pytest.mark.parametrize("operation,initial_visible,initial_reason,"
                             "expected_visible,expected_reason", [
        ("soft_delete", True,  None,           False, "owner_deleted"),
        ("restore",     False, "auto_filtered", True,  None),
    ])
    def test_comment_lifecycle(self, app, test_user, public_model, operation,
                                initial_visible, initial_reason,
                                expected_visible, expected_reason):
        """参数化: 软删除 / 恢复"""
        comment = Comment(
            model_id=public_model.id, user_id=test_user.id,
            content='测试生命周期', is_visible=initial_visible,
            moderation_reason=initial_reason,
        )
        db.session.add(comment)
        db.session.commit()

        if operation == "soft_delete":
            comment.soft_delete("owner_deleted")
        else:
            comment.restore()
        db.session.commit()

        c = db.session.get(Comment, comment.id)
        assert c.is_visible == expected_visible
        assert c.moderation_reason == expected_reason

    def test_is_deleted_by_owner(self, app, test_user, public_model):
        """测试 is_deleted_by_owner 属性"""
        comment = Comment(
            model_id=public_model.id,
            user_id=test_user.id,
            content='测试',
            is_visible=False,
            moderation_reason='owner_deleted',
        )
        db.session.add(comment)
        db.session.commit()
        assert comment.is_deleted_by_owner is True

        comment.moderation_reason = 'auto_filtered'
        db.session.commit()
        assert comment.is_deleted_by_owner is False
