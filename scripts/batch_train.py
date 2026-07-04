"""
批量训练脚本 — 一键训练标准数据集
支持两种执行模式:
  --mode engine : 通过 TrainingService 队列执行 (后台线程池)
  --mode direct : 串行直接训练 (不走线程池, 便于调试)

运行: python scripts/batch_train.py [--mode engine|direct] [--dry-run] [--verbose]
"""
import os
import sys
import json
import time

# ═══════════════════════════════════════════════════════════════
# 使用共享基础设施 (sys.path + argparse)
# ═══════════════════════════════════════════════════════════════
from _common import (
    PROJECT_ROOT, create_base_parser, app_context, STANDARD_JOBS,
    generate_data_aware_params,
)

from app import db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.services.dataset_service import DatasetService
from app.services.training_service import TrainingService


def _get_or_create_dataset(app, admin, cfg: dict) -> Dataset:
    """检查或创建数据集, 返回 Dataset 实例。"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets', cfg['file'])

    if not os.path.exists(file_path):
        raise FileNotFoundError(f'数据集文件不存在: {file_path}')

    ds = Dataset.query.filter_by(file_path=file_path).first()
    if ds:
        print(f'  [OK] 数据集已存在: id={ds.id}')
        return ds

    import uuid as _uuid
    import pandas as pd

    df = pd.read_csv(file_path)
    ds = Dataset(
        name=cfg['file'].replace('_', ' ').replace('.csv', ''),
        description=f'批量训练 - {cfg["task"]} / {cfg["algo"]}',
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        file_format='csv',
        category='tabular',
        is_public=True,
        owner_id=admin.id,
        status='ready',
        uuid=str(_uuid.uuid4()),
        row_count=len(df),
        column_count=len(df.columns),
        summary_json=json.dumps({
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': {col: int(df[col].isna().sum()) for col in df.columns
                               if df[col].isna().sum() > 0},
        }, ensure_ascii=False),
    )
    db.session.add(ds)
    db.session.flush()
    db.session.commit()
    print(f'  [NEW] 数据集已创建: id={ds.id}, {ds.row_count}行 x {ds.column_count}列')
    return ds


def _create_job(admin, ds: Dataset, cfg: dict, app=None):
    """创建训练任务 — 使用数据感知参数。"""
    # 生成数据感知超参数
    if '--no-data-aware' not in sys.argv:
        hyperparams = generate_data_aware_params(
            ds.file_path, cfg['algo'], cfg['task'], cfg['target'],
            framework='sklearn', test_size=0.2, verbose=True,
        )
    else:
        hyperparams = None

    job, error = TrainingService.create_job(
        user=admin,
        name=cfg['name'],
        dataset_id=ds.id,
        description=f'批量训练: {cfg["task"]} / {cfg["algo"]}',
        framework='sklearn',
        total_epochs=1,
        ml_task_type=cfg['task'],
        algorithm=cfg['algo'],
        target_column=cfg['target'],
        test_size=0.2,
        hyperparameters=hyperparams,  # 传入数据感知参数
    )
    return job, error


def _run_engine_mode(app, admin, jobs: list, dry_run: bool = False):
    """通过 TrainingService 队列执行训练 (原 batch_train.py 逻辑)。"""
    from app.executor.engine import get_executor

    results = []
    for i, cfg in enumerate(jobs, 1):
        print(f'\n[{i}/{len(jobs)}] {cfg["name"]}')
        print(f'  文件: {cfg["file"]}  |  {cfg["task"]} / {cfg["algo"]}')

        if dry_run:
            print(f'  [DRY-RUN] 将创建任务并提交到引擎')
            results.append({'name': cfg['name'], 'result': 'dry-run'})
            continue

        try:
            ds = _get_or_create_dataset(app, admin, cfg)
        except FileNotFoundError as e:
            print(f'  [SKIP] {e}')
            results.append({'name': cfg['name'], 'result': '文件不存在'})
            continue

        job, error = _create_job(admin, ds, cfg)
        if error:
            print(f'  [ERROR] 创建任务失败: {error}')
            results.append({'name': cfg['name'], 'result': f'创建失败: {error}'})
            continue

        print(f'  [OK] 任务已创建: id={job.id}')

        success, start_err = TrainingService.start_job(job)
        if not success:
            print(f'  [ERROR] 启动失败: {start_err}')
            results.append({'name': cfg['name'], 'result': f'启动失败: {start_err}'})
            continue

        print(f'  [RUN] 训练已提交...', end=' ')

        timeout = 60
        waited = 0
        while waited < timeout:
            time.sleep(1)
            waited += 1
            db.session.expire_all()
            job = db.session.get(TrainingJob, job.id)
            if job and job.is_finished:
                break

        if job is None:
            print(f'  [ERROR] 训练任务丢失 (可能已被删除)')
            results.append({'name': cfg['name'], 'result': '任务丢失'})
            continue

        if job.is_finished:
            status = job.status
            metrics = json.loads(job.final_metrics_json) if job.final_metrics_json else {}
            acc = metrics.get('test_accuracy') or metrics.get('accuracy', 'N/A')
            if isinstance(acc, float):
                acc = f'{acc:.2%}'
            r2 = metrics.get('test_r2') or metrics.get('r2', 'N/A')
            if isinstance(r2, float):
                r2 = f'{r2:.4f}'

            print(f'{status}  |  Acc={acc}  R²={r2}  |  {job.duration_display}')
            results.append({
                'name': cfg['name'], 'result': status,
                'accuracy': acc, 'r2': r2,
                'duration': job.duration_display,
            })
        else:
            print(f'超时 (仍在 {job.status})')
            results.append({'name': cfg['name'], 'result': f'超时({job.status})'})

    return results


def _run_direct_mode(app, admin, jobs: list, dry_run: bool = False):
    """直接实例化 Trainer 串行训练 (原 batch_train_v2.py 逻辑)。"""
    from app.executor.trainers.sklearn_trainer import SklearnTrainer

    results = []
    for i, cfg in enumerate(jobs, 1):
        t0 = time.time()
        print(f'[{i}/{len(jobs)}] {cfg["name"]} ...', end=' ', flush=True)

        if dry_run:
            print(f'[DRY-RUN] 将直接调用 SklearnTrainer')
            results.append({'name': cfg['name'], 'result': 'dry-run'})
            continue

        try:
            ds = _get_or_create_dataset(app, admin, cfg)
        except FileNotFoundError as e:
            print(f'SKIP: {e}')
            results.append({'name': cfg['name'], 'result': '文件不存在'})
            continue

        job, error = _create_job(admin, ds, cfg)
        if error:
            print(f'FAIL: {error}')
            results.append({'name': cfg['name'], 'result': f'创建失败: {error}'})
            continue

        hyperparams = generate_data_aware_params(
            ds.file_path, cfg['algo'], cfg['task'], cfg['target'],
            framework='sklearn', test_size=0.2, verbose=True,
        )
        trainer = SklearnTrainer(job, ds, hyperparams)

        try:
            trainer.run()
            db.session.expire_all()
            job = db.session.get(TrainingJob, job.id)

            metrics = json.loads(job.final_metrics_json) if job.final_metrics_json else {}
            elapsed = time.time() - t0

            if cfg['task'] == 'classification':
                acc = metrics.get('test_accuracy', 'N/A')
                f1 = metrics.get('test_f1_macro', metrics.get('test_f1_score', 'N/A'))
                score_str = f'Acc={acc:.4f}  F1={f1:.4f}' if isinstance(acc, float) else f'Acc={acc}'
            else:
                r2 = metrics.get('test_r2', 'N/A')
                mse = metrics.get('test_mse', 'N/A')
                score_str = f'R²={r2:.4f}  MSE={mse:.4f}' if isinstance(r2, float) else f'R²={r2}'

            print(f'OK  [{job.status}]  {score_str}  ({elapsed:.1f}s)')
            results.append({
                'name': cfg['name'], 'status': job.status,
                'accuracy': metrics.get('test_accuracy') or metrics.get('accuracy'),
                'f1': metrics.get('test_f1_macro') or metrics.get('test_f1_score') or metrics.get('f1_score'),
                'r2': metrics.get('test_r2') or metrics.get('r2'),
                'mse': metrics.get('test_mse') or metrics.get('mse'),
                'duration': f'{elapsed:.1f}s',
            })

        except Exception as e:
            print(f'ERROR: {e}')
            results.append({'name': cfg['name'], 'result': f'训练异常: {e}'})

    return results


def _print_summary(results: list):
    """打印训练结果汇总表。"""
    print('\n' + '=' * 70)
    n = max(len(r.get('name', '')) for r in results) if results else 20
    print(f'  {"Model":<{n}s} {"Status":<12s} {"Score":>30s} {"Time"}')
    print('=' * 70)
    for r in results:
        name = r.get('name', '?')
        status = r.get('status', r.get('result', '?'))
        acc = r.get('accuracy', '')
        f1 = r.get('f1', '')
        r2 = r.get('r2', '')
        mse = r.get('mse', '')
        if acc and f1:
            score = f'Acc={acc:.4f} / F1={f1:.4f}' if isinstance(acc, float) else str(acc)
        elif r2 and mse:
            score = f'R²={r2:.4f} / MSE={mse:.4f}' if isinstance(r2, float) else str(r2)
        elif acc:
            score = f'Acc={acc}' if not isinstance(acc, float) else f'Acc={acc:.4f}'
        else:
            score = '-'
        dur = r.get('duration', r.get('result', '-'))
        print(f'  {name:<{n}s} {status:<12s} {score:>30s} {dur}')


def main():
    parser = create_base_parser('批量训练标准数据集 (8个)')
    parser.add_argument(
        '--mode', choices=['engine', 'direct'], default='engine',
        help='执行模式: engine=线程池队列 (默认), direct=串行直接训练',
    )
    parser.add_argument(
        '--dataset', help='仅训练指定文件名的数据集 (逗号分隔)',
    )
    parser.add_argument(
        '--output-json', help='结果 JSON 文件路径',
    )
    args = parser.parse_args()

    # 筛选数据集
    jobs = list(STANDARD_JOBS)
    if args.dataset:
        filters = set(args.dataset.split(','))
        jobs = [j for j in jobs if j['file'] in filters]
        if not jobs:
            print(f'[ERROR] 没有匹配的数据集: {args.dataset}')
            print(f'可用: {[j["file"] for j in STANDARD_JOBS]}')
            return

    print(f'批量训练: {len(jobs)} 个数据集, 模式={args.mode}')
    if args.dry_run:
        print('[DRY-RUN] 仅预览, 不执行实际操作\n')

    with app_context('development') as app:
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print('[ERROR] admin 用户不存在，请先运行 seed_data.py')
            return

        print(f'用户: {admin.username} (id={admin.id})')
        print('=' * 60)

        if args.mode == 'direct':
            results = _run_direct_mode(app, admin, jobs, args.dry_run)
        else:
            results = _run_engine_mode(app, admin, jobs, args.dry_run)

        _print_summary(results)

        # 保存汇总 JSON
        output_path = args.output_json
        if not output_path:
            output_path = os.path.join(
                app.config['UPLOAD_FOLDER'], 'datasets', 'training_results.json'
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'\n[OK] 结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
