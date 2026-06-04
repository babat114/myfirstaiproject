"""Train final 9 models: NLP(3) + Other(6)"""
import os, sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.model_record import ModelRecord
from app.services.training_service import TrainingService

CONFIGS = [
    {'model_type': 'nlp', 'dataset_id': 25, 'target_column': 'sentiment',
     'ml_task_type': 'classification',
     'algorithms': ['logistic_regression', 'svm', 'gradient_boosting']},
    {'model_type': 'other', 'dataset_id': 29, 'target_column': 'anomaly_label',
     'ml_task_type': 'classification',
     'algorithms': ['random_forest','logistic_regression','svm','knn','gradient_boosting','decision_tree']},
]

def train_one(app, admin, config, algo, idx):
    mt = config['model_type']
    job_name = f'{algo}-{mt}-{idx}'
    with app.app_context():
        try:
            job, err = TrainingService.create_job(
                user=admin, name=job_name, dataset_id=config['dataset_id'],
                description=f'{mt} model - {algo}', task_type='training',
                framework='sklearn', total_epochs=5, cpu_cores=1, memory_gb=2.0,
                ml_task_type=config['ml_task_type'], algorithm=algo,
                target_column=config['target_column'], test_size=0.2,
                model_type=mt)
            if err:
                print(f'  [FAIL] {err}'); return None
            job = db.session.get(TrainingJob, job.id)
            dataset = db.session.get(Dataset, config['dataset_id'])
            hp = {}
            if job.model and job.model.hyperparameters_json:
                try: hp = json.loads(job.model.hyperparameters_json)
                except: pass
            from app.executor.trainers.sklearn_trainer import SklearnTrainer
            trainer = SklearnTrainer(job, dataset, hp)
            print(f'  [TRAIN] {job_name}...', end=' ', flush=True)
            t0 = time.time(); trainer.run(); elapsed = time.time() - t0
            job = db.session.get(TrainingJob, job.id)
            model = db.session.get(ModelRecord, job.model_id) if job and job.model_id else None
            if job and job.status == 'completed' and model:
                acc = model.accuracy or 0
                print(f'[OK] acc={acc:.4f} ({elapsed:.1f}s)')
                return {'job_id': job.id, 'model_id': model.id, 'name': job_name,
                        'model_type': mt, 'algorithm': algo, 'accuracy': acc,
                        'duration': round(elapsed,1)}
            else:
                print(f'[FAIL] {job.status if job else "?"}: {job.error_message if job else "lost"}')
                return None
        except Exception as e:
            print(f'[ERROR] {str(e)[:120]}')
            import traceback; traceback.print_exc()
            return None

def main():
    app = create_app()
    print('='*50)
    print(f'  Final 9 Models @ {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}')
    print('='*50)
    with app.app_context():
        admin = User.query.filter_by(username='admin').first() or User.query.first()
        for c in CONFIGS:
            ds = db.session.get(Dataset, c['dataset_id'])
            print(f'  [{"OK" if os.path.exists(ds.file_path) else "MISS"}] [{ds.id}] {ds.name}')
    total = sum(len(c['algorithms']) for c in CONFIGS)
    results = []; t0 = time.time(); cnt = 0
    for c in CONFIGS:
        print(f'\n--- {c["model_type"]} ({len(c["algorithms"])} models) ---')
        for i, algo in enumerate(c['algorithms']):
            cnt += 1; print(f'[{cnt}/{total}] {algo}')
            r = train_one(app, admin, c, algo, i+1)
            if r: results.append(r)
    elapsed = time.time() - t0
    print(f'\n{"="*50}')
    print(f'  Done: {len(results)}/{total} in {elapsed/60:.1f} min')
    print(f'{"="*50}')
    rf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'experiments', f'final9_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json')
    os.makedirs(os.path.dirname(rf), exist_ok=True)
    with open(rf,'w',encoding='utf-8') as f:
        json.dump({'ts':datetime.now(timezone.utc).isoformat(),'dur':round(elapsed,1),
                    'ok':len(results),'total':total,'results':results}, f, ensure_ascii=False, indent=2)
    print(f'  Saved: {rf}')

if __name__ == '__main__':
    main()
