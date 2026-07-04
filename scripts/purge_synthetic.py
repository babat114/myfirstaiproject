"""
合成数据集安全清理引擎

分步安全删除合成数据集及其关联模型:
  1. NULLIFY FK引用 (ModelRecord.training_dataset_id, TrainingJob.dataset_id)
  2. DELETE ModelRecord (使用 ModelService.delete_model)
  3. DELETE Dataset (文件 + DB记录)
  4. CLEAN experiment directories

复用 safe_delete_old_models.py 的批量删除模式。

用法:
    python scripts/purge_synthetic.py --dry-run          # 预览
    python scripts/purge_synthetic.py --dry-run --verbose # 详细预览
    python scripts/purge_synthetic.py                     # 执行删除
    python scripts/purge_synthetic.py --force --batch-size 20
"""

import sys
import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob
from app.services.model_service import ModelService


def load_audit_report(audit_path: str = None) -> dict:
    """加载审计报告。"""
    if audit_path is None:
        audit_path = Path(__file__).resolve().parent.parent / 'experiments' / 'dataset_audit_report.json'
    with open(audit_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def nullify_dataset_fks(app, dataset_id: int) -> tuple:
    """清理所有指向该数据集的FK引用。

    Returns (models_nullified, jobs_nullified).
    """
    with app.app_context():
        # 1. Nullify ModelRecord.training_dataset_id
        result1 = db.session.execute(
            db.update(ModelRecord)
            .where(ModelRecord.training_dataset_id == dataset_id)
            .values(training_dataset_id=None)
        )

        # 2. Nullify TrainingJob.dataset_id
        result2 = db.session.execute(
            db.update(TrainingJob)
            .where(TrainingJob.dataset_id == dataset_id)
            .values(dataset_id=None)
        )

        db.session.commit()
        return result1.rowcount, result2.rowcount


def delete_models_for_dataset(app, dataset_id: int, verbose: bool = False) -> tuple:
    """删除所有关联该数据集的模型。

    Returns (deleted, skipped, errors).
    """
    experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'
    deleted = 0
    skipped = 0
    errors = []

    with app.app_context():
        models = ModelRecord.query.filter_by(training_dataset_id=dataset_id).all()

        for model in models:
            try:
                success, error = ModelService.delete_model(model)
                if success:
                    # Clean experiment directory
                    exp_dir = experiments_dir / (model.uuid or '')
                    if exp_dir.exists():
                        try:
                            shutil.rmtree(exp_dir)
                        except Exception as e:
                            if verbose:
                                print(f'    [WARN] 实验目录删除失败 {model.uuid}: {e}')
                    deleted += 1
                    if verbose:
                        print(f'    [OK] Model id={model.id} {model.name}')
                else:
                    skipped += 1
                    errors.append(f'Model {model.id}: {error}')
                    if verbose:
                        print(f'    [FAIL] Model id={model.id}: {error}')
            except Exception as e:
                skipped += 1
                errors.append(f'Model {model.id}: {e}')
                if verbose:
                    print(f'    [ERR] Model id={model.id}: {e}')
                try:
                    db.session.rollback()
                except Exception:
                    pass

        return deleted, skipped, errors


def delete_dataset_safe(app, dataset_id: int, verbose: bool = False) -> tuple:
    """安全删除单个数据集: delete models → nullify remaining FKs → delete file → delete DB record.

    CRITICAL ORDER: Models must be deleted BEFORE nullifying FKs,
    otherwise ModelRecord.training_dataset_id=NULL hides the association.

    Returns (success_bool, error_str).
    """
    with app.app_context():
        ds = db.session.get(Dataset, dataset_id)
        if ds is None:
            return True, None  # already deleted

        try:
            # Step 1: Delete models FIRST (while training_dataset_id still set)
            del_count, skip_count, model_errors = delete_models_for_dataset(
                app, dataset_id, verbose=verbose
            )

            # Step 2: Nullify remaining FK references (post-model-deletion cleanup)
            models_null, jobs_null = nullify_dataset_fks(app, dataset_id)
            if verbose and (models_null > 0 or jobs_null > 0):
                print(f'  FK清理: {models_null} models, {jobs_null} jobs')

            # Step 3: Delete dataset file
            if ds.file_path and os.path.exists(ds.file_path):
                try:
                    os.remove(ds.file_path)
                    if verbose:
                        print(f'  文件已删除: {os.path.basename(ds.file_path)}')
                except Exception as e:
                    if verbose:
                        print(f'  [WARN] 文件删除失败: {e}')

            # Step 4: Delete DB record
            db.session.delete(ds)
            db.session.commit()

            if verbose:
                print(f'  [OK] Dataset id={dataset_id} 已删除 (models={del_count})')

            if model_errors:
                return True, f'Dataset deleted but model errors: {model_errors[:3]}'
            return True, None

        except Exception as e:
            try:
                db.session.rollback()
            except Exception:
                pass
            return False, str(e)


def purge_synthetic(app, audit_report: dict, dry_run: bool = False,
                    force: bool = False, batch_size: int = 10,
                    verbose: bool = False):
    """主清理流程。"""

    # Collect all datasets to delete
    to_delete = []
    for entry in audit_report.get('DELETE_SYNTH', []):
        to_delete.append({
            'dataset_id': entry['dataset_id'],
            'name': entry['name'],
            'filename': entry['filename'],
            'model_count': entry['model_count'],
            'reason': entry['reason'],
        })
    for entry in audit_report.get('DELETE_SYNTH_WITH_MODELS', []):
        to_delete.append({
            'dataset_id': entry['dataset_id'],
            'name': entry['name'],
            'filename': entry['filename'],
            'model_count': entry['model_count'],
            'reason': entry['reason'],
        })

    total_datasets = len(to_delete)
    total_models_to_delete = sum(d['model_count'] for d in to_delete)

    print(f"\n{'=' * 70}")
    print(f"  合成数据集清理引擎")
    print(f"{'=' * 70}")
    print(f"  待删除数据集: {total_datasets}")
    print(f"  待删除模型:   {total_models_to_delete}")
    print(f"  批次大小:     {batch_size}")
    print(f"  模式:         {'DRY-RUN (预览)' if dry_run else '执行删除'}")
    print(f"{'=' * 70}")

    # Preview: show categories
    if dry_run or verbose:
        by_reason = defaultdict(list)
        for d in to_delete:
            # Simplify reason to category
            if 'GEN80' in d['reason']:
                cat = 'GEN80_AUTO'
            elif 'GEN_MULTITYPE' in d['reason'] or '模式匹配' in d['reason']:
                if 'Multiclass' in d['filename'] or 'Binary_' in d['filename']:
                    cat = 'GEN_MULTITYPE'
                elif 'Regression-Large' in d['filename'] or 'Clustering-Blobs' in d['filename'] or 'NLP-Text' in d['filename'] or 'CV-Image' in d['filename'] or 'RL-State' in d['filename'] or 'Gen-Latent' in d['filename'] or 'Other-Anomaly' in d['filename']:
                    cat = 'GEN_MULTITYPE_NAMED'
                else:
                    cat = 'GEN_MULTITYPE_NAMED'
            elif 'real_' in d['reason']:
                cat = 'REAL_FETCH_SYNTH'
            elif 'PUBLIC_SYNTH' in d['reason']:
                cat = 'PUBLIC_SYNTH_DL'
            elif 'UUID' in d['reason']:
                cat = 'UUID_UPLOAD'
            elif 'CATEGORY' in d['reason'] or 'stock_price' in d['filename'] or 'credit_risk' in d['filename']:
                cat = 'CATEGORY_GEN'
            elif 'GENERATIVE' in d['reason']:
                cat = 'GENERATIVE'
            else:
                cat = 'OTHER_SYNTH'
            by_reason[cat].append(d)

        print(f"\n  按类别统计:")
        for cat in sorted(by_reason.keys()):
            items = by_reason[cat]
            models = sum(i['model_count'] for i in items)
            print(f"    {cat:<25s}: {len(items):>4} datasets, {models:>4} models")

    if dry_run:
        print(f"\n  >>> DRY RUN — 不会实际删除 <<<")
        print(f"  运行 'python scripts/purge_synthetic.py --force' 执行删除")
        return 0, 0, []

    # Confirmation
    if not force:
        print(f"\n[!] 此操作将从数据库和磁盘永久删除 {total_datasets} 个数据集及其 {total_models_to_delete} 个模型！")
        confirm = input("输入 'yes' 确认删除，其他任意键取消: ").strip()
        if confirm.lower() != 'yes':
            print("已取消。")
            return 0, 0, []

    # Execute deletion in batches
    start_time = datetime.now()
    deleted_datasets = 0
    deleted_models = 0
    skipped = 0
    all_errors = []
    log_entries = []

    total = len(to_delete)

    for batch_start in range(0, total, batch_size):
        batch = to_delete[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        print(f"\n  === 批次 {batch_num}/{total_batches} ({len(batch)} 个数据集) ===")

        batch_deleted_ds = 0
        batch_deleted_models = 0

        for item in batch:
            ds_id = item['dataset_id']
            ds_name = item['name'][:60]

            if verbose:
                print(f"\n  [{ds_id}] {ds_name} (models={item['model_count']})")

            success, error = delete_dataset_safe(app, ds_id, verbose=verbose)

            if success:
                batch_deleted_ds += 1
                batch_deleted_models += item['model_count']
                if not verbose:
                    print(f"  [OK] DS={ds_id} models={item['model_count']} {item['filename'][:30]}")
                log_entries.append({
                    'dataset_id': ds_id,
                    'name': ds_name,
                    'filename': item['filename'],
                    'model_count': item['model_count'],
                    'action': 'deleted',
                    'reason': item['reason'],
                })
            else:
                skipped += 1
                all_errors.append(f'DS={ds_id}: {error}')
                print(f"  [FAIL] DS={ds_id} {item['filename']}: {error}")
                log_entries.append({
                    'dataset_id': ds_id,
                    'name': ds_name,
                    'filename': item['filename'],
                    'action': 'failed',
                    'error': error,
                })

        deleted_datasets += batch_deleted_ds
        deleted_models += batch_deleted_models

        elapsed = (datetime.now() - start_time).total_seconds()
        progress = min(batch_start + batch_size, total)
        print(f"  批次 {batch_num} 完成: 累计 {deleted_datasets}/{progress} 数据集, "
              f"{deleted_models} 模型, {elapsed:.0f}s")

    # ── Summary ──
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 70}")
    print(f"  清理完成!")
    print(f"{'=' * 70}")
    print(f"  耗时:       {elapsed:.0f}s")
    print(f"  数据集删除: {deleted_datasets}/{total_datasets}")
    print(f"  模型删除:   {deleted_models}")
    print(f"  跳过:       {skipped}")
    if all_errors:
        print(f"  错误:       {len(all_errors)}")
        for e in all_errors[:5]:
            print(f"    - {e}")
    print(f"{'=' * 70}")

    # ── Verify ──
    with app.app_context():
        remaining_ds = Dataset.query.count()
        remaining_models = ModelRecord.query.filter_by(status='trained').count()
        print(f"\n  验证:")
        print(f"    剩余数据集: {remaining_ds}")
        print(f"    剩余trained模型: {remaining_models}")

    # ── Save log ──
    experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'
    experiments_dir.mkdir(exist_ok=True)
    log_path = experiments_dir / f'purge_log_{start_time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'start_time': start_time.isoformat(),
            'elapsed_seconds': elapsed,
            'deleted_datasets': deleted_datasets,
            'deleted_models': deleted_models,
            'skipped': skipped,
            'errors': all_errors,
            'entries': log_entries,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Log] 清理日志已保存: {log_path}")

    return deleted_datasets, deleted_models, all_errors


def main():
    parser = argparse.ArgumentParser(description='合成数据集安全清理引擎')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际删除')
    parser.add_argument('--force', '-f', action='store_true',
                        help='跳过确认提示，直接删除')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='每批删除数据集数量 (默认: 10)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    parser.add_argument('--audit-json', default=None,
                        help='审计报告路径 (默认: experiments/dataset_audit_report.json)')
    args = parser.parse_args()

    # Load audit report
    audit_report = load_audit_report(args.audit_json)

    app = create_app()
    purge_synthetic(
        app, audit_report,
        dry_run=args.dry_run,
        force=args.force,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
