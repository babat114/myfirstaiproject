"""
清理无效训练任务和垃圾数据 (Phase 1)

清理内容:
  1. 删除 failed / cancelled / interrupted 状态的 TrainingJob
  2. 删除超过24小时且从未开始的 queued TrainingJob
  3. 删除空的实验目录
  4. 删除无对应ModelRecord/TrainingJob的孤儿实验目录

Usage:
  python scripts/cleanup_stale_jobs.py --dry-run    # 预览
  python scripts/cleanup_stale_jobs.py --force       # 执行
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.training_job import TrainingJob
from app.models.model_record import ModelRecord
from app.services.training_service import TrainingService


def clean_stale_jobs(app, dry_run=True):
    """清理无效的 TrainingJob 记录"""
    with app.app_context():
        # 1. 找所有 failed / cancelled 的job
        stale_statuses = ['failed', 'cancelled']
        stale_jobs = TrainingJob.query.filter(
            TrainingJob.status.in_(stale_statuses)
        ).all()

        # 2. 找超过24小时的 queued job
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
        stale_queued = TrainingJob.query.filter(
            TrainingJob.status == 'queued',
            TrainingJob.created_at < cutoff
        ).all()

        # 3. 找 interrupted 的job
        interrupted = TrainingJob.query.filter_by(status='interrupted').all()

        all_stale = stale_jobs + stale_queued + interrupted

        # 按状态分组统计
        from collections import Counter
        by_status = Counter(j.status for j in all_stale)

        print(f"{'=' * 60}")
        print(f"  清理无效 TrainingJob")
        print(f"{'=' * 60}")
        for status, count in sorted(by_status.items()):
            print(f"  {status}: {count} 个")
        print(f"  总计: {len(all_stale)} 个")
        print(f"{'=' * 60}")

        if not all_stale:
            print("\n  没有需要清理的无效任务！")
            return 0

        if dry_run:
            print(f"\n  [DRY RUN] 预览完成，加上 --force 实际删除")
            # 显示前几个
            for j in all_stale[:5]:
                print(f"    - [{j.status}] {j.name} (created={j.created_at})")
            if len(all_stale) > 5:
                print(f"    ... 还有 {len(all_stale) - 5} 个")
            return len(all_stale)

        # 实际删除
        deleted = 0
        errors = 0
        for job in all_stale:
            try:
                success, error = TrainingService.delete_job(job)
                if success:
                    deleted += 1
                else:
                    errors += 1
                    print(f"    [FAIL] {job.name}: {error}")
            except Exception as e:
                errors += 1
                print(f"    [ERR] {job.name}: {e}")
                try:
                    db.session.rollback()
                except Exception:
                    pass

        print(f"\n  删除完成: {deleted} 成功, {errors} 失败")
        return deleted


def clean_empty_experiments(app, dry_run=True):
    """清理空实验目录和孤儿目录"""
    experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'

    if not experiments_dir.exists():
        print("  experiments/ 目录不存在")
        return 0, 0

    # 获取所有有效 UUID (ModelRecord + TrainingJob)
    with app.app_context():
        model_uuids = {r[0] for r in db.session.query(ModelRecord.uuid).all()}
        job_uuids = {r[0] for r in db.session.query(TrainingJob.uuid).all()}
    valid_uuids = model_uuids | job_uuids

    all_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
    empty_dirs = [d for d in all_dirs if not any(d.iterdir())]
    orphan_dirs = [d for d in all_dirs if d.name not in valid_uuids and d not in empty_dirs]

    print(f"\n{'=' * 60}")
    print(f"  清理实验目录")
    print(f"{'=' * 60}")
    print(f"  总目录数: {len(all_dirs)}")
    print(f"  空目录: {len(empty_dirs)}")
    print(f"  孤儿目录 (无对应Model/TrainingJob): {len(orphan_dirs)}")
    print(f"{'=' * 60}")

    total_to_delete = len(empty_dirs) + len(orphan_dirs)

    if dry_run:
        print(f"\n  [DRY RUN] 将删除 {total_to_delete} 个目录，加上 --force 实际删除")
        if empty_dirs:
            print(f"\n  空目录 (前10个):")
            for d in empty_dirs[:10]:
                print(f"    - {d.name}")
        if orphan_dirs:
            print(f"\n  孤儿目录 (前10个):")
            for d in orphan_dirs[:10]:
                print(f"    - {d.name}")
        return len(empty_dirs), len(orphan_dirs)

    # 实际删除
    import shutil
    deleted_empty = 0
    deleted_orphan = 0

    for d in empty_dirs:
        try:
            d.rmdir()
            deleted_empty += 1
        except Exception as e:
            print(f"    [ERR] 删除空目录 {d.name}: {e}")

    for d in orphan_dirs:
        try:
            shutil.rmtree(d)
            deleted_orphan += 1
        except Exception as e:
            print(f"    [ERR] 删除孤儿目录 {d.name}: {e}")

    print(f"\n  删除完成: 空目录 {deleted_empty}, 孤儿目录 {deleted_orphan}")
    return deleted_empty, deleted_orphan


def main():
    parser = argparse.ArgumentParser(description='清理无效训练任务和垃圾数据')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='预览模式 (默认)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='实际执行删除')
    parser.add_argument('--skip-jobs', action='store_true',
                        help='跳过 TrainingJob 清理')
    parser.add_argument('--skip-experiments', action='store_true',
                        help='跳过实验目录清理')
    args = parser.parse_args()

    dry_run = not args.force

    if dry_run:
        print(">>> DRY RUN 模式 — 使用 --force 实际执行删除 <<<\n")

    app = create_app()

    total_jobs = 0
    total_empty = 0
    total_orphan = 0

    if not args.skip_jobs:
        total_jobs = clean_stale_jobs(app, dry_run=dry_run)

    if not args.skip_experiments:
        total_empty, total_orphan = clean_empty_experiments(app, dry_run=dry_run)

    print(f"\n{'=' * 60}")
    print(f"  清理汇总")
    print(f"{'=' * 60}")
    print(f"  待清理 TrainingJob: {total_jobs}")
    print(f"  待清理空实验目录: {total_empty}")
    print(f"  待清理孤儿目录: {total_orphan}")
    print(f"  总计: {total_jobs + total_empty + total_orphan}")
    print(f"{'=' * 60}")

    if dry_run and (total_jobs + total_empty + total_orphan) > 0:
        print(f"\n  执行删除: python scripts/cleanup_stale_jobs.py --force")


if __name__ == '__main__':
    main()
