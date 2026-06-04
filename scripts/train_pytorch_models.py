"""
============================================
PyTorch 模型训练 — 修复低准确率 + 框架多样性
NLP(3) + CV(4) + classification(1) + regression(1) + clustering(1) = 10
============================================
"""
import os, sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db

CONFIGS = [
    # === NLP: 替换25%低准确率模型 (200 features, 5 classes) ===
    {'model_type': 'nlp', 'dataset_id': 25, 'target_column': 'sentiment',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [256, 128, 64], 'learning_rate': 0.001,
                     'batch_size': 128, 'dropout': 0.3, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 30, 'name_prefix': 'PyTorch-NLP'},
    {'model_type': 'nlp', 'dataset_id': 25, 'target_column': 'sentiment',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [512, 256, 128, 64], 'learning_rate': 0.0005,
                     'batch_size': 128, 'dropout': 0.4, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 40, 'name_prefix': 'PyTorch-NLP-Deep'},
    {'model_type': 'nlp', 'dataset_id': 25, 'target_column': 'sentiment',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [128, 64, 32], 'learning_rate': 0.003,
                     'batch_size': 64, 'dropout': 0.2, 'weight_decay': 1e-5,
                     'test_size': 0.2},
     'total_epochs': 20, 'name_prefix': 'PyTorch-NLP-Lite'},

    # === CV: 替换29-49%低准确率模型 (150 features, 6 imbalanced classes) ===
    {'model_type': 'computer_vision', 'dataset_id': 26, 'target_column': 'object_class',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [256, 128, 64, 32], 'learning_rate': 0.001,
                     'batch_size': 128, 'dropout': 0.4, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 40, 'name_prefix': 'PyTorch-CV'},
    {'model_type': 'computer_vision', 'dataset_id': 26, 'target_column': 'object_class',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [512, 256, 128, 64, 32], 'learning_rate': 0.0005,
                     'batch_size': 128, 'dropout': 0.5, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 50, 'name_prefix': 'PyTorch-CV-Deep'},
    {'model_type': 'computer_vision', 'dataset_id': 26, 'target_column': 'object_class',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [128, 64], 'learning_rate': 0.003,
                     'batch_size': 64, 'dropout': 0.3, 'weight_decay': 1e-5,
                     'test_size': 0.2},
     'total_epochs': 20, 'name_prefix': 'PyTorch-CV-Lite'},

    # === 为 framework 多样性训练少量其他类型 ===
    {'model_type': 'classification', 'dataset_id': 14, 'target_column': 'target',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [128, 64, 32], 'learning_rate': 0.001,
                     'batch_size': 128, 'dropout': 0.3, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 20, 'name_prefix': 'PyTorch-Class'},
    {'model_type': 'clustering', 'dataset_id': 24, 'target_column': 'cluster_label',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [64, 32], 'learning_rate': 0.001,
                     'batch_size': 128, 'dropout': 0.2, 'weight_decay': 1e-5,
                     'test_size': 0.2},
     'total_epochs': 15, 'name_prefix': 'PyTorch-Cluster'},
    {'model_type': 'reinforcement', 'dataset_id': 27, 'target_column': 'optimal_action',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [128, 64, 32], 'learning_rate': 0.001,
                     'batch_size': 128, 'dropout': 0.3, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 20, 'name_prefix': 'PyTorch-RL'},
    {'model_type': 'generative', 'dataset_id': 28, 'target_column': 'gen_class',
     'ml_task_type': 'classification', 'framework': 'PyTorch',
     'hyperparams': {'hidden_layers': [256, 128, 64], 'learning_rate': 0.0005,
                     'batch_size': 128, 'dropout': 0.3, 'weight_decay': 1e-4,
                     'test_size': 0.2},
     'total_epochs': 25, 'name_prefix': 'PyTorch-Gen'},
]


def train_one(app, admin, cfg, idx):
    from app.models.user import User
    from app.models.dataset import Dataset
    from app.models.training_job import TrainingJob
    from app.models.model_record import ModelRecord
    from app.services.training_service import TrainingService

    hp = cfg['hyperparams'].copy()
    hp.update({'task_type': cfg['ml_task_type'], 'algorithm': 'mlp',
               'target_column': cfg['target_column'], 'test_size': hp.get('test_size', 0.2)})
    name = f"{cfg['name_prefix']}-{idx}"

    with app.app_context():
        try:
            job, err = TrainingService.create_job(
                user=admin, name=name, dataset_id=cfg['dataset_id'],
                description=f'{cfg["model_type"]} PyTorch MLP #{idx}',
                task_type='training', framework=cfg['framework'],
                total_epochs=cfg['total_epochs'], cpu_cores=2, memory_gb=4.0,
                ml_task_type=cfg['ml_task_type'], algorithm='mlp',
                target_column=cfg['target_column'], test_size=hp['test_size'],
                model_type=cfg['model_type'],
                hyperparameters={k: v for k, v in cfg['hyperparams'].items()
                                 if k not in ('task_type', 'algorithm', 'target_column', 'test_size')})
            if err:
                print(f'  [FAIL] create: {err}'); return None

            job = db.session.get(TrainingJob, job.id)
            ds = db.session.get(Dataset, cfg['dataset_id'])

            # Update model hyperparameters with full config
            if job.model:
                all_hp = json.loads(job.model.hyperparameters_json) if job.model.hyperparameters_json else {}
                all_hp.update(hp)
                all_hp.update(cfg['hyperparams'])
                job.model.set_hyperparameters(all_hp)
                db.session.commit()

            from app.executor.trainers.pytorch_trainer import PyTorchTrainer
            trainer = PyTorchTrainer(job, ds, hp)

            print(f'  [TRAIN] {name} (epochs={cfg["total_epochs"]}, '
                  f'hidden={cfg["hyperparams"]["hidden_layers"]})...', end=' ', flush=True)
            t0 = time.time(); trainer.run(); elapsed = time.time() - t0

            job = db.session.get(TrainingJob, job.id)
            model = db.session.get(ModelRecord, job.model_id) if job and job.model_id else None
            if job and job.status == 'completed' and model:
                acc = model.accuracy or 0
                print(f'[OK] acc={acc:.4f} ({elapsed:.0f}s)')
                return {'id': model.id, 'name': name, 'model_type': cfg['model_type'],
                        'accuracy': acc, 'duration': round(elapsed)}
            else:
                st = job.status if job else '?'
                em = (job.error_message or '')[:80]
                print(f'[FAIL] {st}: {em}')
                return None
        except Exception as e:
            print(f'[ERROR] {str(e)[:150]}')
            import traceback; traceback.print_exc()
            return None


def main():
    app = create_app()
    print('='*60)
    print('  PyTorch MLP Training - Fix Low Accuracy + Framework Diversity')
    print(f'  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}')
    print('='*60)

    with app.app_context():
        from app.models.user import User
        admin = User.query.filter_by(username='admin').first() or User.query.first()
        for c in CONFIGS:
            from app.models.dataset import Dataset
            ds = db.session.get(Dataset, c['dataset_id'])
            fe = os.path.exists(ds.file_path) if ds and ds.file_path else False
            print(f'  [{"OK" if fe else "MISS"}] {c["name_prefix"]} -> {c["model_type"]} ({c["framework"]})')

    total = len(CONFIGS)
    results = []; t0 = time.time()
    for i, cfg in enumerate(CONFIGS):
        print(f'\n[{i+1}/{total}] {cfg["name_prefix"]} ({cfg["model_type"]})')
        r = train_one(app, admin, cfg, i+1)
        if r: results.append(r)

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  Done: {len(results)}/{total} PyTorch models in {elapsed/60:.1f} min')
    # Summary by type
    from collections import Counter
    tc = Counter(r['model_type'] for r in results)
    for mt, cnt in tc.most_common():
        mt_results = [r for r in results if r['model_type'] == mt]
        accs = [r['accuracy'] for r in mt_results if r['accuracy']]
        avg_acc = sum(accs)/len(accs) if accs else 0
        print(f'  {mt:20s}: {cnt} models, avg acc={avg_acc:.4f}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
