"""
质量看门狗告警通知模块 v1.0
=======================

支持邮件 (SMTP) 和 Webhook 两种通知渠道。
零外部依赖, 仅使用 Python stdlib (smtplib + urllib)。

Usage::

    from app.utils.notifications import send_watchdog_alert
    result = send_watchdog_alert(report, config=app.config)
    # → {'email_sent': bool, 'webhook_sent': bool, 'suppressed': bool}
"""

import json
import logging
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ── 告警冷却 ──
_cooldowns: dict = {}  # key → last_sent_timestamp
_cooldown_lock = threading.Lock()


def _is_cooldown_active(key: str, cooldown_seconds: int) -> bool:
    """检查是否在冷却期内。Thread-safe。

    Returns True (应抑制告警) 如果距离上次同 key 告警不足 cooldown_seconds。
    同时更新冷却时间戳。
    """
    with _cooldown_lock:
        now = time.time()
        if key in _cooldowns and (now - _cooldowns[key]) < cooldown_seconds:
            return True
        _cooldowns[key] = now
        return False


def send_email(subject: str, body: str, config: dict) -> bool:
    """通过 SMTP + STARTTLS 发送邮件告警 (stdlib only)。

    Args:
        subject: 邮件主题
        body: 邮件正文 (同时作为 text/plain 和 text/html)
        config: Flask app.config 或 dict

    Returns:
        bool — 发送成功返回 True
    """
    smtp_host = config.get('WATCHDOG_SMTP_HOST', '')
    if not smtp_host:
        logger.debug('SMTP host not configured, skipping email alert.')
        return False

    to_emails = [e.strip() for e in config.get('WATCHDOG_SMTP_TO_EMAILS', '').split(',') if e.strip()]
    if not to_emails:
        logger.warning('No recipient emails configured, skipping email alert.')
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.get('WATCHDOG_SMTP_FROM_EMAIL', '')
        msg['To'] = ', '.join(to_emails)

        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        msg.attach(MIMEText(body, 'html', 'utf-8'))

        smtp_port = int(config.get('WATCHDOG_SMTP_PORT', 587))
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(
                config.get('WATCHDOG_SMTP_USER', ''),
                config.get('WATCHDOG_SMTP_PASSWORD', ''),
            )
            server.sendmail(msg['From'], to_emails, msg.as_string())

        logger.info('Watchdog alert email sent to %d recipient(s).', len(to_emails))
        return True

    except Exception as e:
        logger.error('Failed to send watchdog alert email: %s', e)
        return False


def send_webhook(payload: dict, webhook_url: str) -> bool:
    """发送 JSON Webhook 告警 (stdlib urllib)。

    兼容 钉钉/企业微信/飞书/Slack 通用 Incoming Webhook。

    Args:
        payload: JSON-serializable dict
        webhook_url: Webhook endpoint URL

    Returns:
        bool — HTTP 2xx 返回 True
    """
    if not webhook_url:
        logger.debug('Webhook URL not configured, skipping webhook alert.')
        return False

    try:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = Request(
            webhook_url,
            data=data,
            headers={'Content-Type': 'application/json; charset=utf-8'},
        )
        with urlopen(req, timeout=10) as resp:
            status = resp.getcode()
            success = 200 <= status < 300
            if success:
                logger.info('Watchdog webhook alert sent (HTTP %d).', status)
            else:
                logger.warning('Watchdog webhook returned HTTP %d.', status)
            return success

    except URLError as e:
        logger.error('Failed to send watchdog webhook alert: %s', e)
        return False
    except Exception as e:
        logger.error('Failed to send watchdog webhook alert: %s', e)
        return False


def send_watchdog_alert(report: dict, config: dict = None) -> dict:
    """编排看门狗告警: 冷却检查 → 构建消息 → 发送邮件 + Webhook。

    从 report 中提取 summary + alerts, 构建告警消息, 通过已配置的渠道发送。
    所有渠道发送失败均不影响看门狗主循环。

    Args:
        report: WatchdogEngine.scan_all_models() 返回的报告 dict, 含 summary + alerts
        config: Flask app.config 或 dict。为 None 时从 current_app 获取。

    Returns:
        dict — {'email_sent': bool, 'webhook_sent': bool, 'suppressed': bool}
    """
    if config is None:
        try:
            from flask import current_app

            config = current_app.config
        except RuntimeError:
            logger.warning('No app context — cannot send watchdog alert.')
            return {'email_sent': False, 'webhook_sent': False, 'suppressed': False}

    summary = report.get('summary', {})
    n_problem = summary.get('problem_total', 0)
    n_warning = summary.get('warning_total', 0)

    if n_problem == 0 and n_warning == 0:
        logger.debug('Watchdog scan clean — no alert needed.')
        return {'email_sent': False, 'webhook_sent': False, 'suppressed': False}

    # ── 告警冷却 ──
    cooldown = int(config.get('WATCHDOG_ALERT_COOLDOWN', 3600))
    cooldown_key = 'watchdog_problem' if n_problem > 0 else 'watchdog_warning'

    if _is_cooldown_active(cooldown_key, cooldown):
        logger.info(
            'Watchdog alert suppressed (cooldown active, %ds since last %s alert).',
            cooldown,
            cooldown_key,
        )
        return {'email_sent': False, 'webhook_sent': False, 'suppressed': True}

    # ── 构建告警消息 ──
    alerts = report.get('alerts', [])
    subject = f'[AI Platform Watchdog] {n_problem} problem(s), {n_warning} warning(s) detected'

    lines = [
        'AI Platform — Model Quality Watchdog Report',
        f'Generated: {report.get("generated_at", "unknown")}',
        '',
        'Summary:',
        f'  Total scanned:          {summary.get("scanned", 0)}',
        f'  Tier 1 (definite_delete): {summary.get("definite_delete", 0)}',
        f'  Tier 2 (high_risk):       {summary.get("high_risk_delete", 0)}',
        f'  Tier 3 (warning_only):    {summary.get("warning_only", 0)}',
        f'  Healthy:                  {summary.get("healthy", 0)}',
        '',
    ]
    if alerts:
        lines.append('Problem Models:')
        for alert in alerts[:10]:
            lines.append(f'  - {alert}')
        if len(alerts) > 10:
            lines.append(f'  ... and {len(alerts) - 10} more')
    else:
        lines.append('No critical problems detected.')

    body = '\n'.join(lines)

    # ── 发送 ──
    email_sent = send_email(subject, body, config)

    webhook_payload = {
        'msgtype': 'text',
        'text': {'content': f'{subject}\n\n{body}'},
        'source': 'ai-platform-watchdog',
    }
    webhook_url = config.get('WATCHDOG_WEBHOOK_URL', '')
    webhook_sent = send_webhook(webhook_payload, webhook_url)

    return {
        'email_sent': email_sent,
        'webhook_sent': webhook_sent,
        'suppressed': False,
    }
