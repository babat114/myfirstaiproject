"""
============================================
TensorFlow + ONNX + Other 框架模型批量训练
每种类型 ~3模型 × 6+类型 = 20+ 模型
============================================
"""
import os, sys, json, time
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db

# ===================================================================
# 训练配置
# ===================================================================
CONFIGS = [
    # === TensorFlow/Keras 模型 ===
    # NLP (使用 v5 修正数据)
    {'mt':'nlp','ds':25,'tc':'sentiment','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[256,128,64],'lr':0.001,'bs':128,'drop':0.3,'wd':1e-4},
     'epochs':30,'name':'TF-NLP'},
    {'mt':'nlp','ds':25,'tc':'sentiment','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[512,256,128,64],'lr':0.0005,'bs':128,'drop':0.4,'wd':1e-4},
     'epochs':40,'name':'TF-NLP-Deep'},
    # CV
    {'mt':'computer_vision','ds':26,'tc':'object_class','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[256,128,64,32],'lr':0.001,'bs':128,'drop':0.4,'wd':1e-4},
     'epochs':40,'name':'TF-CV'},
    {'mt':'computer_vision','ds':26,'tc':'object_class','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[512,256,128,64,32],'lr':0.0005,'bs':128,'drop':0.5,'wd':1e-4},
     'epochs':50,'name':'TF-CV-Deep'},
    # Classification
    {'mt':'classification','ds':14,'tc':'target','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[128,64,32],'lr':0.001,'bs':128,'drop':0.3,'wd':1e-4},
     'epochs':20,'name':'TF-Class'},
    # Reinforcement
    {'mt':'reinforcement','ds':27,'tc':'optimal_action','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[128,64,32],'lr':0.001,'bs':128,'drop':0.3,'wd':1e-4},
     'epochs':20,'name':'TF-RL'},
    # Generative
    {'mt':'generative','ds':28,'tc':'gen_class','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[256,128,64],'lr':0.001,'bs':128,'drop':0.3,'wd':1e-4},
     'epochs':25,'name':'TF-Gen'},
    # Other
    {'mt':'other','ds':29,'tc':'anomaly_label','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[128,64,32],'lr':0.001,'bs':128,'drop':0.3,'wd':1e-4},
     'epochs':20,'name':'TF-Other'},
    # Regression
    {'mt':'regression','ds':22,'tc':'target','task':'regression','fw':'TensorFlow',
     'hp':{'hidden_layers':[256,128,64,32],'lr':0.001,'bs':128,'drop':0.2,'wd':1e-4},
     'epochs':50,'name':'TF-Reg'},
    # Clustering
    {'mt':'clustering','ds':24,'tc':'cluster_label','task':'classification','fw':'TensorFlow',
     'hp':{'hidden_layers':[64,32],'lr':0.001,'bs':128,'drop':0.2,'wd':1e-5},
     'epochs':15,'name':'TF-Cluster'},

    # === ONNX 模型 (sklearn训练 + ONNX转换) ===
    {'mt':'nlp','ds':25,'tc':'sentiment','task':'classification','fw':'ONNX',
     'algo':'random_forest','epochs':5,'name':'ONNX-NLP-RF'},
    {'mt':'computer_vision','ds':26,'tc':'object_class','task':'classification','fw':'ONNX',
     'algo':'random_forest','epochs':5,'name':'ONNX-CV-RF'},
    {'mt':'classification','ds':14,'tc':'target','task':'classification','fw':'ONNX',
     'algo':'gradient_boosting','epochs':5,'name':'ONNX-Class-GB'},
    {'mt':'reinforcement','ds':27,'tc':'optimal_action','task':'classification','fw':'ONNX',
     'algo':'logistic_regression','epochs':5,'name':'ONNX-RL-LR'},
    {'mt':'generative','ds':28,'tc':'gen_class','task':'classification','fw':'ONNX',
     'algo':'random_forest','epochs':5,'name':'ONNX-Gen-RF'},
    {'mt':'other','ds':29,'tc':'anomaly_label','task':'classification','fw':'ONNX',
     'algo':'decision_tree','epochs':5,'name':'ONNX-Other-DT'},
    {'mt':'clustering','ds':24,'tc':'cluster_label','task':'classification','fw':'ONNX',
     'algo':'knn','epochs':5,'name':'ONNX-Cluster-KNN'},

    # === Other 框架模型 (sklearn训练, 不同算法) ===
    {'mt':'nlp','ds':25,'tc':'sentiment','task':'classification','fw':'Other',
     'algo':'decision_tree','epochs':5,'name':'Other-NLP-DT'},
    {'mt':'computer_vision','ds':26,'tc':'object_class','task':'classification','fw':'Other',
     'algo':'gradient_boosting','epochs':5,'name':'Other-CV-GB'},
    {'mt':'reinforcement','ds':27,'tc':'optimal_action','task':'classification','fw':'Other',
     'algo':'decision_tree','epochs':5,'name':'Other-RL-DT'},
    {'mt':'generative','ds':28,'tc':'gen_class','task':'classification','fw':'Other',
     'algo':'logistic_regression','epochs':5,'name':'Other-Gen-LR'},
    {'mt':'other','ds':29,'tc':'anomaly_label','task':'classification','fw':'Other',
     'algo':'logistic_regression','epochs':5,'name':'Other-Other-LR'},
    {'mt':'regression','ds':22,'tc':'target','task':'regression','fw':'Other',
     'algo':'ridge','epochs':5,'name':'Other-Reg-Ridge'},
]


def train_one(app, admin, cfg, idx):
    from app.models.user import User
    from app.models.dataset import Dataset
    from app.models.training_job import TrainingJob
    from app.models.model_record import ModelRecord
    from app.services.training_service import TrainingService

    is_sklearn = cfg['fw'] in ('ONNX', 'Other') or 'algo' in cfg
    is_tf = cfg['fw'] == 'TensorFlow'

    if is_sklearn:
        algo = cfg.get('algo', 'random_forest')
    else:
        algo = 'mlp'

    hp_all = {}
    if is_tf:
        hp_all = dict(cfg['hp'])
        hp_all.update({'task_type': cfg['task'], 'algorithm': algo,
                       'target_column': cfg['tc'], 'test_size': 0.2})
    elif is_sklearn:
        hp_all = {'task_type': cfg['task'], 'algorithm': algo,
                  'target_column': cfg['tc'], 'test_size': 0.2}

    with app.app_context():
        try:
            job, err = TrainingService.create_job(
                user=admin, name=cfg['name'], dataset_id=cfg['ds'],
                description=f'{cfg["mt"]} {cfg["fw"]} model',
                task_type='training', framework=cfg['fw'],
                total_epochs=cfg['epochs'], cpu_cores=2, memory_gb=4.0,
                ml_task_type=cfg['task'], algorithm=algo,
                target_column=cfg['tc'], test_size=0.2, model_type=cfg['mt'],
                hyperparameters={k: v for k, v in cfg.get('hp', {}).items()
                                 if k not in ('task_type', 'algorithm', 'target_column', 'test_size')}
                if is_tf else None)
            if err:
                print(f'  [FAIL] create: {err}'); return None

            job = db.session.get(TrainingJob, job.id)
            ds = db.session.get(Dataset, cfg['ds'])

            if job.model:
                all_hp = json.loads(job.model.hyperparameters_json) if job.model.hyperparameters_json else {}
                all_hp.update(hp_all)
                if is_tf:
                    all_hp.update(cfg['hp'])
                job.model.set_hyperparameters(all_hp)
                db.session.commit()

            # 选择训练器
            if is_tf:
                from app.executor.trainers.keras_trainer import KerasTrainer
                trainer = KerasTrainer(job, ds, hp_all)
            else:
                from app.executor.trainers.sklearn_trainer import SklearnTrainer
                trainer = SklearnTrainer(job, ds, hp_all)

            print(f'  [TRAIN] {cfg["name"]} ({cfg["fw"]})...', end=' ', flush=True)
            t0 = time.time(); trainer.run(); elapsed = time.time() - t0

            job = db.session.get(TrainingJob, job.id)
            model = db.session.get(ModelRecord, job.model_id) if job and job.model_id else None

            if job and job.status == 'completed' and model:
                # ONNX 模型转换
                if cfg['fw'] == 'ONNX' and model.model_file_path:
                    try:
                        _convert_to_onnx(model)
                    except Exception as e:
                        pass  # ONNX 转换失败不阻塞

                acc = model.accuracy or 0
                if cfg['task'] == 'regression':
                    r2 = model.metrics_dict.get('test_r2', 0)
                    print(f'[OK] R2={r2:.4f} ({elapsed:.0f}s)')
                else:
                    print(f'[OK] acc={acc:.4f} ({elapsed:.0f}s)')
                return {'id': model.id, 'name': cfg['name'], 'fw': cfg['fw'],
                        'accuracy': acc, 'duration': round(elapsed)}
            else:
                st = job.status if job else '?'
                em = (job.error_message or '')[:80]
                print(f'[FAIL] {st}: {em}')
                return None
        except Exception as e:
            print(f'[ERROR] {str(e)[:120]}')
            import traceback; traceback.print_exc()
            return None


def _convert_to_onnx(model):
    """将 sklearn 模型转换为 ONNX 格式"""
    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        model_path = model.model_file_path
        if not model_path or not os.path.exists(model_path):
            return

        import pickle
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        sk_model = data.get('model')
        if sk_model is None:
            return

        n_features = len(data.get('feature_names', []))
        if n_features == 0:
            return

        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onx = convert_sklearn(sk_model, initial_types=initial_type)

        onnx_path = model_path.replace('.pkl', '.onnx')
        onnx.save_model(onx, onnx_path)

        # 更新模型记录
        from app import db
        model.weights_file_path = onnx_path
        model.file_size = (model.file_size or 0) + os.path.getsize(onnx_path)
        db.session.commit()
    except Exception as e:
        pass  # 静默失败，不阻塞训练流程


def main():
    app = create_app()
    print('='*60)
    print('  TF + ONNX + Other Framework Training')
    print(f'  {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}')
    print('='*60)

    with app.app_context():
        from app.models.user import User
        from app.models.dataset import Dataset
        admin = User.query.filter_by(username='admin').first() or User.query.first()
        for c in CONFIGS:
            ds = db.session.get(Dataset, c['ds'])
            fe = os.path.exists(ds.file_path) if ds and ds.file_path else False
            print(f'  [{"OK" if fe else "MISS"}] {c["name"]:25s} -> {c["mt"]:20s} ({c["fw"]})')

    total = len(CONFIGS)
    results = []
    t0 = time.time()
    for i, cfg in enumerate(CONFIGS):
        print(f'\n[{i+1}/{total}] {cfg["name"]} ({cfg["fw"]})')
        r = train_one(app, admin, cfg, i + 1)
        if r:
            results.append(r)

    elapsed = time.time() - t0
    print(f'\n{"="*60}')
    print(f'  Results: {len(results)}/{total} models in {elapsed/60:.1f} min')
    from collections import Counter
    fw_counts = Counter(r['fw'] for r in results)
    for fw, cnt in fw_counts.most_common():
        fw_results = [r for r in results if r['fw'] == fw]
        accs = [r['accuracy'] for r in fw_results if r['accuracy'] > 0]
        print(f'  {fw:15s}: {cnt} models, avg acc={sum(accs)/len(accs):.4f}' if accs else f'  {fw:15s}: {cnt} models')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
