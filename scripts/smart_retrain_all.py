"""
智能批量重训练 — 使用真实数据集 + 数据感知参数

读取 fetch_real_datasets.py 采集的真实数据集 (real_*.csv)，
为每个算法-数据集组合生成数据感知参数并训练。

与 batch_train.py 的区别:
  - 使用真实世界数据集 (非标准toy数据集)
  - 每个参数都经过 ParameterGuidanceService 数据感知优化
  - 训练完成后自动捕获实际 sklearn 参数到 DB
  - 可选 GridSearchCV 调优 (--auto-tune)

Usage:
    python scripts/smart_retrain_all.py --dry-run --verbose
    python scripts/smart_retrain_all.py --mode direct --limit 5
    python scripts/smart_retrain_all.py --auto-tune --algo random_forest
    python scripts/smart_retrain_all.py --dataset german_credit
"""

import os
import sys
import json
import time
import io
import re

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import (
    PROJECT_ROOT, create_base_parser, app_context, setup_verbose,
    generate_data_aware_params,
)

# ═══════════════════════════════════════════════════════════════
# 训练配置生成
# ═══════════════════════════════════════════════════════════════

def _detect_real_datasets(dataset_dir: str) -> list:
    """扫描 uploads/datasets/ 中的 real_*.csv 数据集。"""
    if not os.path.isdir(dataset_dir):
        return []

    datasets = []
    for fname in sorted(os.listdir(dataset_dir)):
        if fname.startswith('real_') and fname.endswith('.csv'):
            filepath = os.path.join(dataset_dir, fname)
            key = fname.replace('real_', '').replace('.csv', '')
            datasets.append({
                'key': key,
                'filename': fname,
                'filepath': filepath,
            })
    return datasets


def _build_training_configs(dataset_dir: str, algo_filter: str = None,
                             dataset_filter: str = None) -> list:
    """根据可用数据集和算法清单生成训练配置。

    从 fetch_real_datasets.py 复用 DATASET_MANIFEST 获取算法映射。
    """
    from fetch_real_datasets import DATASET_MANIFEST

    # 构建 key → manifest 的映射
    manifest_map = {m['key']: m for m in DATASET_MANIFEST}

    # 扫描可用数据集
    available = _detect_real_datasets(dataset_dir)

    if dataset_filter:
        filters = set(dataset_filter.split(','))
        available = [d for d in available if d['key'] in filters]

    configs = []
    for ds_info in available:
        manifest = manifest_map.get(ds_info['key'])
        if not manifest:
            continue  # 未知数据集，跳过

        for algo in manifest.get('algorithms', []):
            if algo_filter and algo != algo_filter:
                continue

            # 检测目标列（可能被 fetch 脚本更正过）
            target_col = manifest.get('target', '')
            display_name = manifest.get('display_name', ds_info['key'])

            configs.append({
                'key': ds_info['key'],
                'filename': ds_info['filename'],
                'filepath': ds_info['filepath'],
                'display_name': display_name,
                'algorithm': algo,
                'task_type': manifest.get('category', 'classification'),
                'target_column': target_col,
                'domain': manifest.get('domain', ''),
                'description': manifest.get('description', ''),
            })

    return configs


# ═══════════════════════════════════════════════════════════════
# 训练执行
# ═══════════════════════════════════════════════════════════════

def _run_engine_mode(app, admin, configs: list, dry_run: bool = False,
                     auto_tune: bool = False, verbose: bool = False,
                     skip_existing: bool = False):
    """通过 TrainingService 队列执行训练。"""
    from app import db
    from app.models.dataset import Dataset
    from app.models.training_job import TrainingJob
    from app.services.training_service import TrainingService

    results = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        print(f'\n[{i}/{total}] {cfg["display_name"]} — {cfg["algorithm"]}')
        print(f'  文件: {cfg["filename"]}  |  {cfg["task_type"]}  |  {cfg["domain"]}')

        # 检测 MLP → 使用 PyTorch trainer
        is_mlp = cfg['algorithm'] == 'mlp'
        framework = 'pytorch' if is_mlp else 'sklearn'
        total_epochs = 10 if is_mlp else 1

        if dry_run:
            print(f'  [DRY-RUN] 将创建任务并提交到引擎')
            results.append({'config': cfg, 'result': 'dry-run'})
            continue

        # 1. 查找或创建数据集记录
        ds = Dataset.query.filter_by(file_path=cfg['filepath']).first()
        if not ds:
            import pandas as pd
            import uuid as _uuid
            df = pd.read_csv(cfg['filepath'])
            ds = Dataset(
                name=f'真实数据集: {cfg["display_name"]}',
                description=cfg.get('description', f'真实{cfg["domain"]}数据集'),
                file_path=cfg['filepath'],
                file_size=os.path.getsize(cfg['filepath']),
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
                }, ensure_ascii=False),
            )
            db.session.add(ds)
            db.session.flush()
            db.session.commit()
            print(f'  [NEW] 数据集已创建: id={ds.id}')

        # 检查是否已有训练模型 (--skip-existing)
        if skip_existing:
            from app.models.model_record import ModelRecord
            existing_models = ModelRecord.query.filter_by(
                training_dataset_id=ds.id, status='trained'
            ).all()
            existing_match = None
            for m in existing_models:
                if m.hyperparameters_dict.get('algorithm') == cfg['algorithm']:
                    existing_match = m
                    break
            if existing_match:
                print(f'  [SKIP] 已有训练模型 id={existing_match.id} '
                      f'(ds={ds.id}, algo={cfg["algorithm"]})')
                results.append({'config': cfg, 'result': 'skipped',
                                'reason': 'existing_model'})
                continue

        # 2. 生成数据感知参数
        hyperparams = generate_data_aware_params(
            cfg['filepath'], cfg['algorithm'], cfg['task_type'],
            cfg['target_column'], framework=framework, test_size=0.2,
            verbose=verbose,
        )

        # 3. 创建训练任务
        if auto_tune:
            # 使用 AutoML 模式
            from app.services.hyperparameter_tuning import HyperparameterTuningService
            job, error = HyperparameterTuningService.create_tuned_training(
                user=admin,
                name=f'{cfg["display_name"]}-{cfg["algorithm"]}',
                dataset_id=ds.id,
                task_type=cfg['task_type'],
                test_size=0.2,
                cv_folds=3,
                random_state=None,
            )
        else:
            job, error = TrainingService.create_job(
                user=admin,
                name=f'{cfg["display_name"]}-{cfg["algorithm"]}',
                dataset_id=ds.id,
                description=f'数据感知训练: {cfg["domain"]} / {cfg["algorithm"]}',
                framework=framework,
                total_epochs=total_epochs,
                ml_task_type=cfg['task_type'],
                algorithm=cfg['algorithm'],
                target_column=cfg['target_column'],
                test_size=0.2,
                hyperparameters=hyperparams,
            )

        if error:
            print(f'  [ERROR] 创建任务失败: {error}')
            results.append({'config': cfg, 'result': f'创建失败: {error}'})
            continue

        print(f'  [OK] 任务已创建: id={job.id}')

        # 4. 启动训练
        success, start_err = TrainingService.start_job(job)
        if not success:
            print(f'  [ERROR] 启动失败: {start_err}')
            results.append({'config': cfg, 'result': f'启动失败: {start_err}'})
            continue

        # 5. 等待完成
        if is_mlp:
            wait_timeout = 300  # PyTorch MLP 训练更慢
        elif auto_tune:
            wait_timeout = 300  # AutoML 需要更长时间
        else:
            wait_timeout = 90

        waited = 0
        while waited < wait_timeout:
            time.sleep(1)
            waited += 1
            db.session.expire_all()
            job = db.session.get(TrainingJob, job.id)
            if job and job.is_finished:
                break

        if job is None:
            print(f'  [ERROR] 训练任务丢失')
            results.append({'config': cfg, 'result': '任务丢失'})
            continue

        if job.is_finished:
            status = job.status
            metrics = json.loads(job.final_metrics_json) if job.final_metrics_json else {}
            score_str = ''
            if cfg['task_type'] == 'classification':
                acc = metrics.get('test_accuracy') or metrics.get('accuracy')
                f1 = metrics.get('test_f1_macro') or metrics.get('test_f1_score') or metrics.get('f1_score')
                if isinstance(acc, float):
                    score_str = f'Acc={acc:.4f}'
                if isinstance(f1, float):
                    score_str += f'  F1={f1:.4f}'
            elif cfg['task_type'] == 'regression':
                r2 = metrics.get('test_r2') or metrics.get('r2')
                mse = metrics.get('test_mse') or metrics.get('mse')
                if isinstance(r2, float):
                    score_str = f'R²={r2:.4f}'
                if isinstance(mse, float):
                    score_str += f'  MSE={mse:.4f}'
            else:
                silhouette = metrics.get('silhouette_score')
                if isinstance(silhouette, float):
                    score_str = f'Silhouette={silhouette:.4f}'

            print(f'  [{status.upper()}] {score_str}  |  {job.duration_display}')
            results.append({
                'config': cfg, 'result': status,
                'metrics': metrics,
                'param_source': job.model.hyperparameters_dict.get('param_source', 'unknown')
                if job.model else 'unknown',
                'duration': job.duration_display,
            })
        else:
            print(f'  超时 (仍在 {job.status})')
            results.append({'config': cfg, 'result': f'超时({job.status})'})

    return results


def _run_direct_mode(app, admin, configs: list, dry_run: bool = False,
                     auto_tune: bool = False, verbose: bool = False,
                     skip_existing: bool = False):
    """直接实例化 Trainer 串行训练 (不走线程池)。"""
    from app import db
    from app.models.dataset import Dataset
    from app.models.training_job import TrainingJob
    from app.services.training_service import TrainingService
    from app.executor.trainers.sklearn_trainer import SklearnTrainer
    from app.executor.trainers.pytorch_trainer import PyTorchTrainer

    results = []
    total = len(configs)

    for i, cfg in enumerate(configs, 1):
        t0 = time.time()
        print(f'\n[{i}/{total}] {cfg["display_name"]} — {cfg["algorithm"]}')
        print(f'  文件: {cfg["filename"]}  |  {cfg["task_type"]}  |  {cfg["domain"]}')

        # 检测 MLP → 使用 PyTorch trainer
        is_mlp = cfg['algorithm'] == 'mlp'
        framework = 'pytorch' if is_mlp else 'sklearn'
        total_epochs = 10 if is_mlp else 1

        if dry_run:
            print(f'  [DRY-RUN] 将直接调用 SklearnTrainer')
            results.append({'config': cfg, 'result': 'dry-run'})
            continue

        # 1. 查找或创建数据集
        ds = Dataset.query.filter_by(file_path=cfg['filepath']).first()
        if not ds:
            import pandas as pd
            import uuid as _uuid
            df = pd.read_csv(cfg['filepath'])
            ds = Dataset(
                name=f'真实数据集: {cfg["display_name"]}',
                description=cfg.get('description', f'真实{cfg["domain"]}数据集'),
                file_path=cfg['filepath'],
                file_size=os.path.getsize(cfg['filepath']),
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
                }, ensure_ascii=False),
            )
            db.session.add(ds)
            db.session.flush()
            db.session.commit()
            print(f'  [NEW] 数据集已创建: id={ds.id}')

        # 检查是否已有训练模型 (--skip-existing)
        if skip_existing:
            from app.models.model_record import ModelRecord
            existing_models = ModelRecord.query.filter_by(
                training_dataset_id=ds.id, status='trained'
            ).all()
            existing_match = None
            for m in existing_models:
                if m.hyperparameters_dict.get('algorithm') == cfg['algorithm']:
                    existing_match = m
                    break
            if existing_match:
                print(f'  [SKIP] 已有训练模型 id={existing_match.id} '
                      f'(ds={ds.id}, algo={cfg["algorithm"]})')
                results.append({'config': cfg, 'result': 'skipped',
                                'reason': 'existing_model'})
                continue

        # 2. 生成数据感知参数
        hyperparams = generate_data_aware_params(
            cfg['filepath'], cfg['algorithm'], cfg['task_type'],
            cfg['target_column'], framework=framework, test_size=0.2,
            verbose=verbose,
        )

        # 3. 创建训练任务
        job, error = TrainingService.create_job(
            user=admin,
            name=f'{cfg["display_name"]}-{cfg["algorithm"]}',
            dataset_id=ds.id,
            description=f'数据感知训练: {cfg["domain"]} / {cfg["algorithm"]}',
            framework=framework,
            total_epochs=total_epochs,
            ml_task_type=cfg['task_type'],
            algorithm=cfg['algorithm'],
            target_column=cfg['target_column'],
            test_size=0.2,
            hyperparameters=hyperparams,
        )
        if error:
            print(f'  [ERROR] 创建任务失败: {error}')
            results.append({'config': cfg, 'result': f'创建失败: {error}'})
            continue

        # 4. 直接训练 (MLP → PyTorch, 其他 → sklearn)
        if is_mlp:
            trainer = PyTorchTrainer(job, ds, hyperparams)
        else:
            trainer = SklearnTrainer(job, ds, hyperparams)
        try:
            trainer.run()
            db.session.expire_all()
            job = db.session.get(TrainingJob, job.id)

            metrics = json.loads(job.final_metrics_json) if job.final_metrics_json else {}
            elapsed = time.time() - t0

            score_parts = []
            if cfg['task_type'] == 'classification':
                acc = metrics.get('test_accuracy') or metrics.get('accuracy')
                f1 = metrics.get('test_f1_macro') or metrics.get('test_f1_score') or metrics.get('f1_score')
                if isinstance(acc, float):
                    score_parts.append(f'Acc={acc:.4f}')
                if isinstance(f1, float):
                    score_parts.append(f'F1={f1:.4f}')
            elif cfg['task_type'] == 'regression':
                r2 = metrics.get('test_r2') or metrics.get('r2')
                mse = metrics.get('test_mse') or metrics.get('mse')
                if isinstance(r2, float):
                    score_parts.append(f'R²={r2:.4f}')
                if isinstance(mse, float):
                    score_parts.append(f'MSE={mse:.4f}')
            else:
                sil = metrics.get('silhouette_score')
                if isinstance(sil, float):
                    score_parts.append(f'Silhouette={sil:.4f}')

            param_source = 'unknown'
            if job.model:
                hp_dict = job.model.hyperparameters_dict
                param_source = hp_dict.get('param_source', 'unknown')

            print(f'  [OK] [{job.status}] {" ".join(score_parts)}  '
                  f'param_source={param_source}  ({elapsed:.1f}s)')
            results.append({
                'config': cfg,
                'status': job.status,
                'metrics': metrics,
                'param_source': param_source,
                'duration': f'{elapsed:.1f}s',
            })

        except Exception as e:
            print(f'  [ERROR] 训练异常: {e}')
            import traceback
            if verbose:
                traceback.print_exc()
            results.append({'config': cfg, 'result': f'训练异常: {e}'})

    return results


def _print_summary(results: list):
    """打印训练结果汇总表。"""
    print('\n' + '=' * 80)
    success = sum(1 for r in results if r.get('result') == 'completed' or r.get('status') == 'completed')
    failed = sum(1 for r in results if '失败' in str(r.get('result', '')) or r.get('status') == 'failed')
    dry = sum(1 for r in results if r.get('result') == 'dry-run')

    print(f'  总计: {len(results)}  |  成功: {success}  |  失败: {failed}  |  预览: {dry}')
    print('=' * 80)

    # 按算法分组统计
    algo_stats = {}
    for r in results:
        cfg = r.get('config', {})
        algo = cfg.get('algorithm', '?')
        if algo not in algo_stats:
            algo_stats[algo] = {'total': 0, 'success': 0}
        algo_stats[algo]['total'] += 1
        if r.get('result') == 'completed' or r.get('status') == 'completed':
            algo_stats[algo]['success'] += 1

    if algo_stats:
        print('\n按算法统计:')
        for algo in sorted(algo_stats.keys()):
            s = algo_stats[algo]
            print(f'  {algo:<35s} {s["success"]}/{s["total"]}')


def main():
    parser = create_base_parser('智能批量重训练 — 真实数据集 + 数据感知参数')
    parser.add_argument(
        '--mode', choices=['engine', 'direct'], default='direct',
        help='执行模式: engine=线程池队列, direct=串行直接训练 (default: direct)',
    )
    parser.add_argument(
        '--algo', help='仅训练特定算法 (如 random_forest)',
    )
    parser.add_argument(
        '--dataset', help='仅训练特定数据集 (逗号分隔，如 german_credit,adult)',
    )
    parser.add_argument(
        '--limit', type=int, default=0,
        help='最多训练数量 (0=全部)',
    )
    parser.add_argument(
        '--auto-tune', action='store_true', default=False,
        help='启用 AutoML 超参数调优 (GridSearchCV)',
    )
    parser.add_argument(
        '--skip-existing', action='store_true', default=False,
        help='跳过已有训练模型的 (dataset_id + algorithm) 组合',
    )
    parser.add_argument(
        '--output-json', help='结果 JSON 文件路径',
    )
    args = parser.parse_args()
    setup_verbose(args)

    with app_context('development') as app:
        from app.models.user import User

        admin = User.query.filter_by(username='admin').first()
        if not admin:
            print('[ERROR] admin 用户不存在，请先运行 seed_data.py')
            return

        dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'datasets')

        # 构建训练配置
        configs = _build_training_configs(
            dataset_dir,
            algo_filter=args.algo,
            dataset_filter=args.dataset,
        )

        if args.limit and args.limit > 0:
            configs = configs[:args.limit]

        if not configs:
            print('[ERROR] 没有匹配的训练配置。')
            print(f'  检查: {dataset_dir}/real_*.csv 是否存在')
            print(f'  提示: 先运行 python scripts/fetch_real_datasets.py 采集数据集')
            return

        print(f'智能重训练: {len(configs)} 个任务')
        print(f'  模式: {args.mode}  |  AutoML: {args.auto_tune}')
        print(f'  用户: {admin.username}')
        if args.dry_run:
            print(f'  [DRY-RUN] 仅预览\n')
        print('=' * 60)

        # 执行训练
        if args.mode == 'direct':
            results = _run_direct_mode(
                app, admin, configs,
                dry_run=args.dry_run,
                auto_tune=args.auto_tune,
                verbose=args.verbose,
                skip_existing=args.skip_existing,
            )
        else:
            results = _run_engine_mode(
                app, admin, configs,
                dry_run=args.dry_run,
                auto_tune=args.auto_tune,
                verbose=args.verbose,
                skip_existing=args.skip_existing,
            )

        _print_summary(results)

        # 保存汇总 JSON
        if not args.dry_run:
            output_path = args.output_json or os.path.join(
                dataset_dir, 'smart_retrain_results.json'
            )
            # 清理不可序列化的对象
            clean_results = []
            for r in results:
                cr = {k: v for k, v in r.items()
                      if k != 'config' or isinstance(v, dict)}
                if 'config' in r:
                    cr['key'] = r['config'].get('key', '')
                    cr['algorithm'] = r['config'].get('algorithm', '')
                    cr['display_name'] = r['config'].get('display_name', '')
                clean_results.append(cr)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(clean_results, f, ensure_ascii=False, indent=2)
            print(f'\n[OK] 结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
