"""
内部时区工具 — 独立模块，不参与 app.utils.__init__ 的导入链 (避免循环导入)

模型文件 (app.models.*) 应直接从此模块导入 localnow，
因为 app.utils.__init__ 会触发 decorators → services → models 的导入链。
"""
from datetime import datetime, timezone, timedelta

# 北京时区 (UTC+8)
_BEIJING_TZ = timezone(timedelta(hours=8))


def localnow():
    """返回北京时间 (UTC+8) 的 naive datetime，适配 MySQL DATETIME 列 (不存时区)

    所有需要写入数据库的时间戳都应使用此函数，
    避免 datetime.now(timezone.utc) 导致的 8 小时时差问题。
    """
    return datetime.now(_BEIJING_TZ).replace(tzinfo=None)
