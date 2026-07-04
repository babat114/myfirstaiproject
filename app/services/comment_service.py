"""
============================================
评论服务
评论的增删查 + 自动内容审核
============================================
"""
import re
from typing import Optional, Tuple, List
from sqlalchemy.orm import joinedload
from app import db, logger
from app.utils.helpers import paginate_query
from app.models.comment import Comment
from app.models.model_record import ModelRecord
from app.models.user import User
from app._timezone import localnow


# ============ 内容审核关键词库 ============
# 命中以下任一模式 → 自动屏蔽评论 (is_visible=False)

_BANNED_PATTERNS: List[str] = [
    # 英文攻击性词汇
    # 无歧义的词用子串匹配 (捕获 fuckyou/shithead/bullshit 等拼接形式)
    r'fuck',           # 无合法英文单词包含 "fuck"
    r'cunt',           # 无合法英文单词包含 "cunt"
    r'nigger', r'nigga',  # 种族歧视
    r'faggot',         # 恐同
    # 有极小歧义的词用词边界 (避免误伤 shitake/damning/assassin 等)
    r'\b(shit|shitty|shithead|shitstorm|shitbag|shitface|bullshit|dipshit)\b',
    r'\b(damn|damned|damnit|goddamn)\b',
    r'\b(bitch|bitches|bitching|bitchy|sonofabitch)\b',
    r'\b(ass|asshole|asshat|jackass|dumbass|smartass|badass)\b',
    r'\b(bastard|bastards)\b',
    r'\b(dick|dickhead|dicks)\b',
    r'\b(whore|whores|whoring)\b',
    r'\b(slut|sluts|slutty)\b',
    r'\b(piss|pissed|pissing)\b',
    r'\b(chink|chinks)\b',
    r'\b(retard|retards|retarded)\b',
    # 中文敏感词 — 人身攻击 / 辱骂
    r'傻逼', r'蠢货', r'白痴', r'脑残', r'智障',
    r'去死', r'滚蛋', r'垃圾',
    r'草泥马', r'操你', r'日你', r'艹',
    r'他妈的', r'你妈的', r'你妈', r'他妈', r'特么', r'你大爷',
    r'狗日的', r'王八蛋', r'龟儿子',
    # 拼音/缩写绕过
    r'\bcnm\b', r'\bnmsl\b', r'\bsb\b', r'\btmd\b', r'\bmlgb\b',
    r'\bwcnm\b', r'\bcnmb\b',
    # 色情/低俗
    r'裸聊', r'约炮', r'一夜情', r'性交', r'做爱',
    r'黄色', r'色情', r'av\b', r'成人电影',
    # 赌博/诈骗
    r'赌博', r'博彩', r'彩票.*加', r'赌场',
    r'兼职.*日结', r'刷单', r'返利.*联系',
    # 广告/spam 模式
    r'加微信', r'加Q\w*:', r'加群', r'联系Q\w*:',
    r'http[s]?://(?!([^/]*\.)?(github|gitee|arxiv|kaggle|huggingface|pytorch|tensorflow)\.)',
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}.*(?:联系|合作|推广|广告)',
]

# 编译正则 (忽略大小写)
_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in _BANNED_PATTERNS
]


class CommentService:
    """评论管理服务 — 含自动内容审核"""

    # ============ 内容审核 ============

    @staticmethod
    def _moderate_content(content: str) -> Tuple[bool, Optional[str]]:
        """
        审核评论内容

        Returns:
            (is_clean, matched_pattern_or_None)
        """
        if not content or not content.strip():
            return False, 'empty_content'

        for pattern in _COMPILED_PATTERNS:
            match = pattern.search(content)
            if match:
                matched_text = match.group()
                logger.info(f'评论审核: 命中违规模式 "{pattern.pattern}" → "{matched_text}"')
                return False, f'auto_filtered:{pattern.pattern[:50]}'

        return True, None

    @staticmethod
    def _sanitize_content(content: str) -> str:
        """清理评论内容: 移除HTML标签, 限制长度"""
        # 移除 HTML 标签
        cleaned = re.sub(r'<[^>]*>', '', content)
        # 限制长度 (最多2000字符)
        cleaned = cleaned.strip()[:2000]
        return cleaned

    # ============ CRUD ============

    @staticmethod
    def add_comment(
        user: User,
        model_id: int,
        content: str,
        parent_id: int = None,
    ) -> Tuple[Optional[Comment], Optional[str]]:
        """
        添加评论 (含自动审核)

        Returns:
            (Comment, error_message) — Comment 可能为 None (审核不通过)
        """
        try:
            # 验证模型存在且可评论 (仅公开模型允许评论)
            model = db.session.get(ModelRecord, model_id)
            if not model:
                return None, '模型不存在。'
            if not model.is_public:
                return None, '仅公开模型支持评论。'
            if not model.is_viewable_by(user):
                return None, '您没有权限评论此模型。'

            # 如果是回复, 验证父评论存在
            if parent_id:
                parent = db.session.get(Comment, parent_id)
                if not parent or not parent.is_visible:
                    return None, '要回复的评论不存在。'
                if parent.model_id != model_id:
                    return None, '回复的评论不属于此模型。'

            # 清理内容
            content = CommentService._sanitize_content(content)
            if not content:
                return None, '评论内容不能为空。'

            # 内容审核
            is_clean, reason = CommentService._moderate_content(content)

            comment = Comment(
                model_id=model_id,
                user_id=user.id,
                content=content,
                parent_id=parent_id,
                is_visible=is_clean,
                moderation_reason=reason if not is_clean else None,
            )
            db.session.add(comment)
            db.session.commit()

            if not is_clean:
                logger.warning(
                    f'评论被自动屏蔽: user={user.username}, '
                    f'model={model_id}, reason={reason}'
                )
                return comment, '您的评论包含不当内容，已被系统自动屏蔽。如有疑问请联系管理员。'

            logger.info(f'评论发表成功: user={user.username}, model={model_id}')
            return comment, None

        except Exception as e:
            db.session.rollback()
            from app.utils.helpers import sanitize_service_error
            return None, sanitize_service_error(e, '添加评论失败')

    @staticmethod
    def delete_comment(
        comment_id: int,
        user: User,
        permanent: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        删除评论

        Args:
            comment_id: 评论ID
            user: 当前用户
            permanent: True=物理删除(管理员), False=软删除(标记不可见)

        Returns:
            (success, error_message)
        """
        try:
            comment = db.session.get(Comment, comment_id)
            if not comment:
                return False, '评论不存在。'

            # 权限检查: 作者 或 管理员
            if comment.user_id != user.id and not user.is_admin:
                return False, '您没有权限删除此评论。'

            if permanent and user.is_admin:
                # 管理员物理删除
                db.session.delete(comment)
                db.session.commit()
                logger.info(f'管理员 {user.username} 物理删除了评论 {comment_id}')
            else:
                # 软删除
                reason = 'admin_removed' if user.is_admin else 'owner_deleted'
                comment.soft_delete(reason)
                db.session.commit()
                logger.info(f'评论 {comment_id} 被软删除 (by {user.username}, reason={reason})')

            return True, None

        except Exception as e:
            db.session.rollback()
            from app.utils.helpers import sanitize_service_error
            return False, sanitize_service_error(e, '删除评论失败')

    @staticmethod
    def restore_comment(
        comment_id: int,
        user: User,
    ) -> Tuple[bool, Optional[str]]:
        """管理员恢复被屏蔽的评论"""
        if not user.is_admin:
            return False, '需要管理员权限。'

        try:
            comment = db.session.get(Comment, comment_id)
            if not comment:
                return False, '评论不存在。'
            if comment.is_visible:
                return False, '评论未被屏蔽。'

            comment.restore()
            db.session.commit()
            logger.info(f'管理员 {user.username} 恢复了评论 {comment_id}')
            return True, None

        except Exception as e:
            db.session.rollback()
            from app.utils.helpers import sanitize_service_error
            return False, sanitize_service_error(e, '恢复评论失败')

    @staticmethod
    def get_comments_for_model(
        model_id: int,
        user: Optional[User] = None,
        page: int = 1,
        per_page: int = 20,
        include_hidden: bool = False,
    ) -> dict:
        """
        获取模型的评论列表 (分页, 仅顶级评论, 子评论嵌套)

        Args:
            model_id: 模型ID
            user: 当前用户 (决定是否显示被屏蔽的评论)
            include_hidden: 是否包含被屏蔽的评论 (仅管理员)

        Returns:
            分页结果字典
        """
        query = Comment.query.options(
            joinedload(Comment.user),  # 预加载用户信息, 避免 N+1
        ).filter_by(
            model_id=model_id,
            parent_id=None,  # 仅顶级评论
        )

        if not include_hidden:
            # 普通用户只看到可见评论
            if user and user.is_admin:
                # 管理员可以看到被屏蔽的评论 (但标记出来)
                pass
            else:
                # 普通用户: 仅可见评论
                query = query.filter_by(is_visible=True)
        else:
            if not (user and user.is_admin):
                query = query.filter_by(is_visible=True)

        query = query.order_by(Comment.created_at.desc())

        return paginate_query(query, page, per_page, item_key='comments', transform_fn=lambda x: x.to_dict())

    @staticmethod
    def get_replies_for_comment(
        parent_id: int,
        user: Optional[User] = None,
    ) -> list:
        """获取某条评论的所有可见回复"""
        from sqlalchemy import select
        stmt = select(Comment).where(Comment.parent_id == parent_id)

        if not (user and user.is_admin):
            stmt = stmt.where(Comment.is_visible == True)

        stmt = stmt.order_by(Comment.created_at.asc())
        return [c.to_dict() for c in db.session.execute(stmt).scalars().all()]

    @staticmethod
    def get_comment_by_id(comment_id: int) -> Optional[Comment]:
        """根据ID获取评论"""
        return db.session.get(Comment, comment_id)
