"""
============================================
模型质量自动监控看门狗 (Model Quality Watchdog)
============================================
定期扫描所有已训练模型, 检测质量退化, 生成告警报告。

检测维度 (复用 diagnose_model_quality.py):
  1. 常数预测器 — 对所有输入预测同一类别
  2. 零方差概率 — predict_proba 方差为 0
  3. 完美准确率 (acc=1.0) — 可证明过拟合
  4. 近完美准确率 (acc>=0.999) — 高风险
  5. F1-Accuracy 差距 — 类别不平衡信号
  6. 训练-测试差距 — train/test split 过拟合信号
  7. 健康评分 — ParameterGuidanceService 综合评估

Tier 分类 (与 diagnose/clean 脚本兼容):
  Tier 1 (definite_delete): 常数预测器 or acc=1.0
  Tier 2 (high_risk_delete): 近完美/大gap + acc>=0.95
  Tier 3 (warning_only): 低健康评分 or 其他警告

运行模式:
  --once          单次扫描 + 生成报告
  --daemon        后台守护进程, 按 --interval 定时扫描
  --quick         仅做快速预筛选 (常数预测器检查, 不加载参数指导服务)

Usage:
  python scripts/quality_watchdog.py --once
  python scripts/quality_watchdog.py --once --output-json experiments/watchdog_report.json
  python scripts/quality_watchdog.py --once --auto-clean --force    # CAUTION: 自动删除 Tier 1
  python scripts/quality_watchdog.py --daemon --interval 360        # 每 6 小时扫描
  python scripts/quality_watchdog.py --quick                        # 仅快速检查
"""
import sys
import os
import json
import argparse
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.services.model_service import ModelService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Watchdog Engine
# ═══════════════════════════════════════════════════════════════

class WatchdogEngine:
    """模型质量监控引擎 — 单次扫描、增量检测、报告生成。

    复用 diagnose_model_quality.py 的诊断函数, 避免重复实现。
    """

    def __init__(self, app, health_threshold: int = 40,
                 quick_mode: bool = False):
        self.app = app
        self.health_threshold = health_threshold
        self.quick_mode = quick_mode
        self._diagnose_fn = None  # lazy import

    def _get_diagnose_fn(self):
        """延迟导入诊断函数 (避免循环导入)。"""
        if self._diagnose_fn is None:
            from scripts.diagnose_model_quality import (
                diagnose_model, detect_constant_predictor,
                DEFAULT_HEALTH_THRESHOLD,
            )
            self._diagnose_fn = diagnose_model
            self._detect_cp_fn = detect_constant_predictor
            self._DEFAULT_HEALTH = DEFAULT_HEALTH_THRESHOLD
        return self._diagnose_fn

    def scan_all_models(self, since: Optional[datetime] = None,
                        status_filter: str = 'trained') -> dict:
        """扫描所有 (或自 since 以来) 已训练模型。

        Args:
            since: 仅扫描此时间之后创建/更新的模型 (增量模式)
            status_filter: 模型状态过滤 (默认 'trained')

        Returns:
            WatchdogReport dict:
              {generated_at, config, summary: {total, tier1, tier2, tier3, healthy},
               models: [...], alerts: [...]}
        """
        diagnose = self._get_diagnose_fn()

        with self.app.app_context():
            query = ModelRecord.query.filter_by(status=status_filter)
            if since is not None:
                query = query.filter(ModelRecord.updated_at >= since)
            models = query.order_by(ModelRecord.created_at.desc()).all()

            if not models:
                logger.info('No models to scan.')
                return self._empty_report()

            total = len(models)
            logger.info(
                f'Scanning {total} models '
                f'(quick={self.quick_mode}, since={since or "all"})...'
            )

            results = []
            tier_counts = {'definite_delete': 0, 'high_risk_delete': 0,
                           'warning_only': 0, 'healthy': 0}
            alerts = []

            for i, m in enumerate(models):
                try:
                    if self.quick_mode:
                        result = self._quick_diagnose(m)
                    else:
                        result = diagnose(
                            m, inspect_model=True,
                            health_threshold=self.health_threshold
                        )
                    results.append(result)

                    tier = result.get('tier', 'healthy')
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                    if tier in ('definite_delete', 'high_risk_delete'):
                        alerts.append(self._format_alert(result))

                    if (i + 1) % 50 == 0:
                        logger.info(f'  Progress: {i+1}/{total} models scanned')

                except Exception as e:
                    logger.warning(
                        f'  [SKIP] model id={m.id} "{m.name}": {e}'
                    )
                    results.append({
                        'model_uuid': m.uuid,
                        'model_id': m.id,
                        'model_name': m.name,
                        'error': str(e),
                        'tier': 'error',
                    })

        # ── 汇总 ──
        n_problem = tier_counts['definite_delete'] + tier_counts['high_risk_delete']
        n_warning = tier_counts['warning_only']
        report = {
            'generated_at': datetime.now().isoformat(),
            'config': {
                'health_threshold': self.health_threshold,
                'quick_mode': self.quick_mode,
                'since': since.isoformat() if since else None,
                'status_filter': status_filter,
            },
            'summary': {
                'total': total,
                'scanned': len(results),
                'definite_delete': tier_counts['definite_delete'],
                'high_risk_delete': tier_counts['high_risk_delete'],
                'warning_only': tier_counts['warning_only'],
                'healthy': tier_counts['healthy'],
                'problem_total': n_problem,
                'warning_total': n_warning,
            },
            'alerts': alerts,
            'models': results,
        }
        return report

    def _quick_diagnose(self, model: ModelRecord) -> dict:
        """快速诊断 — 仅检查常数预测器 (不调用参数指导服务)。

        For large model counts where full health_score computation is too slow.
        """
        from scripts.diagnose_model_quality import (
            detect_constant_predictor,
            PERFECT_ACC_THRESHOLD, NEAR_PERFECT_THRESHOLD,
        )
        reasons = []
        warnings_list = []
        details = {}

        acc = model.accuracy

        # 完美准确率
        if acc is not None and acc == PERFECT_ACC_THRESHOLD:
            reasons.append('perfect_accuracy')
            details['accuracy'] = acc

        # 近完美准确率
        if acc is not None and PERFECT_ACC_THRESHOLD > acc >= NEAR_PERFECT_THRESHOLD:
            reasons.append('near_perfect_accuracy')
            details['accuracy'] = acc

        # 常数预测器
        if model.model_file_path:
            is_broken, cp_details = detect_constant_predictor(model)
            details['constant_predictor'] = cp_details
            if is_broken:
                reasons.append('constant_or_zero_var')
        else:
            details['constant_predictor'] = {'checked': False}

        # Tier 判定
        definite = {'constant_or_zero_var', 'perfect_accuracy'}
        high_risk = {'near_perfect_accuracy'}
        has_definite = any(r in definite for r in reasons)

        if has_definite:
            tier = 'definite_delete'
        elif any(r in high_risk for r in reasons) and acc is not None and acc >= 0.95:
            tier = 'high_risk_delete'
        elif reasons:
            tier = 'warning_only'
        else:
            tier = 'healthy'

        return {
            'model_uuid': model.uuid,
            'model_id': model.id,
            'model_name': model.name,
            'model_type': model.model_type,
            'framework': model.framework,
            'status': model.status,
            'accuracy': acc,
            'precision': model.precision,
            'recall': model.recall,
            'f1_score': model.f1_score,
            'dataset_name': model.training_dataset.name if model.training_dataset else None,
            'tier': tier,
            'reasons': reasons,
            'warnings': warnings_list,
            'details': details,
        }

    def auto_clean(self, report: dict, tiers: tuple = ('definite_delete',),
                   dry_run: bool = True) -> dict:
        """根据报告自动删除问题模型。

        Args:
            report: scan_all_models() 返回的报告
            tiers: 要删除的 Tier 列表 (默认仅 Tier 1)
            dry_run: True=仅预览, False=实际删除

        Returns:
            {deleted: int, skipped: int, errors: int, details: [...]}
        """
        to_delete = [
            m for m in report['models']
            if m.get('tier') in tiers and m.get('model_id')
        ]
        if not to_delete:
            logger.info('No models to clean.')
            return {'deleted': 0, 'skipped': 0, 'errors': 0, 'details': []}

        logger.info(
            f'[AUTO-CLEAN] {"DRY-RUN" if dry_run else "LIVE"}: '
            f'{len(to_delete)} models in tiers {tiers}'
        )

        deleted = 0
        skipped = 0
        errors = 0
        details = []

        for entry in to_delete:
            model_id = entry['model_id']
            model_name = entry.get('model_name', '?')
            tier = entry.get('tier', '?')

            if dry_run:
                logger.info(f'  [DRY-RUN] Would delete: "{model_name}" (id={model_id}, tier={tier})')
                deleted += 1
                details.append({
                    'model_id': model_id, 'model_name': model_name,
                    'tier': tier, 'action': 'would_delete',
                })
                continue

            with self.app.app_context():
                model = db.session.get(ModelRecord, model_id)
                if model is None:
                    skipped += 1
                    details.append({
                        'model_id': model_id, 'model_name': model_name,
                        'tier': tier, 'action': 'skipped (not found)',
                    })
                    continue

                success, error = ModelService.delete_model(model)
                if success:
                    deleted += 1
                    logger.info(f'  [DELETED] "{model_name}" (id={model_id}, tier={tier})')
                    details.append({
                        'model_id': model_id, 'model_name': model_name,
                        'tier': tier, 'action': 'deleted',
                    })
                else:
                    errors += 1
                    logger.error(f'  [ERROR] "{model_name}" (id={model_id}): {error}')
                    details.append({
                        'model_id': model_id, 'model_name': model_name,
                        'tier': tier, 'action': f'error: {error}',
                    })

        return {
            'deleted': deleted, 'skipped': skipped, 'errors': errors,
            'details': details,
        }

    @staticmethod
    def _format_alert(result: dict) -> str:
        """格式化单个告警消息。"""
        name = result.get('model_name', '?')
        tier = result.get('tier', '?')
        acc = result.get('accuracy')
        reasons = result.get('reasons', [])
        acc_str = f'acc={acc:.4f}' if acc is not None else 'acc=N/A'
        return (
            f'[{tier.upper()}] {name} ({acc_str}) — '
            f'Reasons: {", ".join(reasons) if reasons else "none"}'
        )

    @staticmethod
    def _empty_report() -> dict:
        return {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total': 0, 'scanned': 0,
                'definite_delete': 0, 'high_risk_delete': 0,
                'warning_only': 0, 'healthy': 0,
                'problem_total': 0, 'warning_total': 0,
            },
            'alerts': [],
            'models': [],
        }


# ═══════════════════════════════════════════════════════════════
# Daemon Mode
# ═══════════════════════════════════════════════════════════════

def _start_daemon(engine: WatchdogEngine, interval_minutes: int,
                  output_dir: str = 'experiments', config: dict = None):
    """启动后台守护线程, 定期扫描模型质量。

    Uses the same threading.Event.wait() pattern as app/utils/cache.py
    """
    stop_event = threading.Event()
    interval_seconds = interval_minutes * 60

    def _watchdog_loop():
        logger.info(
            f'Watchdog daemon started — scanning every {interval_minutes} min '
            f'({interval_seconds}s)'
        )
        # 首次立即扫描
        try:
            report = engine.scan_all_models()
            _save_report(report, output_dir)
            _print_summary(report)
            _try_send_alert(report, config)
        except Exception as e:
            logger.error(f'Initial scan failed: {e}')

        while not stop_event.wait(interval_seconds):
            try:
                report = engine.scan_all_models()
                _save_report(report, output_dir)
                n_problem = report['summary'].get('problem_total', 0)
                if n_problem > 0:
                    logger.warning(
                        f'Watchdog: {n_problem} problem models detected. '
                        f'Check {output_dir}/watchdog_report.json'
                    )
                    for alert in report.get('alerts', []):
                        logger.warning(f'  {alert}')
                else:
                    logger.info(f'Watchdog: all {report["summary"]["total"]} models healthy')
                _try_send_alert(report, config)
            except Exception as e:
                logger.error(f'Watchdog scan failed: {e}')

    thread = threading.Thread(
        target=_watchdog_loop, daemon=True, name='quality-watchdog'
    )
    thread.start()
    logger.info(f'Watchdog thread started (daemon=True, interval={interval_minutes}min)')
    return thread, stop_event


def _save_report(report: dict, output_dir: str):
    """保存报告到 JSON 文件 (带时间戳)。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(output_dir, f'watchdog_report_{ts}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # 同时更新 latest
    latest_path = os.path.join(output_dir, 'watchdog_report.json')
    with open(latest_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _print_summary(report: dict):
    """打印报告摘要到日志。"""
    s = report['summary']
    logger.info(
        f'Watchdog Report: {s["scanned"]}/{s["total"]} scanned | '
        f'Tier1={s["definite_delete"]} Tier2={s["high_risk_delete"]} '
        f'Tier3={s["warning_only"]} Healthy={s["healthy"]}'
    )
    alerts = report.get('alerts', [])
    if alerts:
        for alert in alerts[:5]:  # 最多打印前 5 条
            logger.warning(f'  {alert}')
        if len(alerts) > 5:
            logger.warning(f'  ... and {len(alerts) - 5} more alerts')


def _try_send_alert(report: dict, config: dict = None):
    """尝试发送看门狗告警通知 (v1.0)。

    失败不抛异常 — 通知系统独立于看门狗主循环, 任何失败都不应影响扫描。
    """
    try:
        from app.utils.notifications import send_watchdog_alert
        result = send_watchdog_alert(report, config=config)
        if result.get('email_sent'):
            logger.info('Watchdog alert email sent.')
        if result.get('webhook_sent'):
            logger.info('Watchdog alert webhook sent.')
        if result.get('suppressed'):
            logger.info('Watchdog alert suppressed (cooldown active).')
    except Exception as e:
        logger.error(f'Failed to send watchdog notification: {e}')


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='模型质量自动监控看门狗',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python scripts/quality_watchdog.py --once                         # 单次扫描
  python scripts/quality_watchdog.py --once --quick                 # 快速扫描 (仅常数预测器)
  python scripts/quality_watchdog.py --once --auto-clean --force    # 扫描并自动删除 Tier 1
  python scripts/quality_watchdog.py --daemon --interval 360        # 每6小时守护进程
  python scripts/quality_watchdog.py --once --since 2026-06-25      # 仅检查6/25后更新的模型
        ''',
    )
    # ── 运行模式 ──
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true',
                      help='单次扫描 + 生成报告')
    mode.add_argument('--daemon', action='store_true',
                      help='后台守护进程, 按 --interval 定时扫描')

    # ── 扫描参数 ──
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: 仅检测常数预测器+完美准确率 (不调用参数指导服务)')
    parser.add_argument('--since', type=str, default=None,
                        help='仅扫描此日期后更新的模型 (ISO格式: 2026-06-25)')
    parser.add_argument('--status', type=str, default='trained',
                        help='模型状态过滤 (默认: trained)')
    parser.add_argument('--alert-threshold', type=int, default=40,
                        help='健康评分告警阈值 (默认: 40, 仅非quick模式)')
    parser.add_argument('--tier', type=str, default='1,2',
                        help='告警的 Tier 级别 (默认: 1,2)')

    # ── 自动清理 (危险) ──
    parser.add_argument('--auto-clean', action='store_true',
                        help='自动删除 Tier 1 问题模型 (需要 --force)')
    parser.add_argument('--force', action='store_true',
                        help='确认执行自动清理 (配合 --auto-clean)')

    # ── 守护进程参数 ──
    parser.add_argument('--interval', type=int, default=1440,
                        help='守护进程扫描间隔, 单位分钟 (默认: 1440 = 24h)')

    # ── 输出 ──
    parser.add_argument('--output-json', type=str, default=None,
                        help='单次扫描报告输出路径 (守护模式自动写入 experiments/)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='报告输出目录 (默认: experiments)')
    parser.add_argument('--quiet', action='store_true',
                        help='减少日志输出')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # ── 参数校验 ──
    if args.daemon and not args.once and not args.daemon:
        # 默认 --once
        args.once = True

    if args.auto_clean and not args.force:
        logger.error(
            '--auto-clean requires --force confirmation. '
            'This is a DESTRUCTIVE operation.'
        )
        sys.exit(1)

    if args.daemon and args.auto_clean:
        logger.error(
            '--auto-clean is not supported in --daemon mode for safety. '
            'Run --once --auto-clean --force manually to clean models.'
        )
        sys.exit(1)

    # ── 解析 since 日期 ──
    since_dt = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error(f'Invalid --since date: {args.since}. Use ISO format: 2026-06-25')
            sys.exit(1)

    # ── 解析 tier ──
    tier_set = set()
    tier_map = {'1': 'definite_delete', '2': 'high_risk_delete', '3': 'warning_only'}
    for t in args.tier.split(','):
        t = t.strip()
        if t in tier_map:
            tier_set.add(tier_map[t])

    # ── 创建应用 ──
    app = create_app()

    # ── 创建引擎 ──
    engine = WatchdogEngine(
        app,
        health_threshold=args.alert_threshold if not args.quick else 40,
        quick_mode=args.quick,
    )

    # ── Daemon 模式 ──
    if args.daemon:
        logger.info(f'Starting watchdog daemon (interval={args.interval}min)...')
        thread, stop_event = _start_daemon(
            engine, args.interval, args.output_dir,
            config=app.config,
        )
        try:
            while thread.is_alive():
                thread.join(1)
        except KeyboardInterrupt:
            logger.info('Shutting down watchdog daemon...')
            stop_event.set()
            thread.join(timeout=10)
            logger.info('Watchdog stopped.')
        return

    # ── Once 模式 ──
    logger.info('Starting one-shot watchdog scan...')
    report = engine.scan_all_models(since=since_dt, status_filter=args.status)

    # ── 输出报告 ──
    output_path = args.output_json
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f'Report saved to: {output_path}')
    else:
        _save_report(report, args.output_dir)
        logger.info(f'Report saved to: {args.output_dir}/watchdog_report.json')

    _print_summary(report)

    # ── 发送告警通知 (v1.0) ──
    _try_send_alert(report, app.config)

    # ── 自动清理 ──
    if args.auto_clean:
        clean_tiers = tuple(tier_set) if tier_set else ('definite_delete',)
        clean_result = engine.auto_clean(report, tiers=clean_tiers, dry_run=False)
        logger.info(
            f'Auto-clean result: {clean_result["deleted"]} deleted, '
            f'{clean_result["skipped"]} skipped, {clean_result["errors"]} errors'
        )

    # ── 非零退出码 ──
    problem_count = report['summary'].get('problem_total', 0)
    if problem_count > 0:
        logger.warning(
            f'Watchdog found {problem_count} problem models. '
            f'Review the report and run clean_overfit_models.py to delete them.'
        )
        # exit code 1 if problems found (useful for CI/CD)
        sys.exit(1)
    else:
        logger.info('All models healthy — no quality issues detected.')


if __name__ == '__main__':
    main()
