
"""Retrain Car Evaluation + Mushroom with anti-overfitting measures."""
import argparse, json, os, sys, time, logging, numpy as np, pandas as pd
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app, db
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord
from app.models.user import User

DATASET_IDS = [147, 134]
ANTI_OVERFIT = {
    'random_forest': {'algorithm': 'random_forest', 'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight': 'balanced', 'random_state': 42},
    'gradient_boosting': {'algorithm': 'gradient_boosting', 'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 20, 'min_samples_leaf': 10, 'learning_rate': 0.05, 'random_state': 42},
    'decision_tree': {'algorithm': 'decision_tree', 'max_depth': 6, 'min_samples_split': 15, 'min_samples_leaf': 8, 'class_weight': 'balanced', 'random_state': 42},
}

def retrain(ds_id, dry_run=False):
    app = create_app()
    with app.app_context():
        ds = db.session.get(Dataset, ds_id)
        if not ds: return []
        logger.info(f'Retraining: {ds.name}')
        admin = User.query.filter_by(username='admin').first() or User.query.first()
        results = []
        for algo, params in ANTI_OVERFIT.items():
            logger.info(f'  {algo}...')
            if dry_run: results.append({'algo': algo, 'status': 'dry_run'}); continue
            try:
                df = pd.read_csv(ds.file_path); tc = df.columns[-1]
                X = df.drop(columns=[tc]); y = df[tc]
                from sklearn.model_selection import cross_val_score, StratifiedKFold
                from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder(); ye = le.fit_transform(y.astype(str))
                if algo == 'random_forest':
                    m = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], class_weight=params['class_weight'], random_state=42, n_jobs=-1)
                elif algo == 'gradient_boosting':
                    m = GradientBoostingClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], learning_rate=params['learning_rate'], random_state=42)
                else:
                    m = DecisionTreeClassifier(max_depth=params['max_depth'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'], class_weight=params['class_weight'], random_state=42)
                cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cvs = cross_val_score(m, X, ye, cv=cv5, scoring='accuracy')
                train_acc = cvs.mean()
                logger.info(f'    CV: {train_acc:.4f} +/- {cvs.std():.4f}')
                m.fit(X, ye)
                ind_ds = Dataset.query.filter_by(source_dataset_id=ds.id, is_test_set=True).first()
                ind_acc = None
                if ind_ds:
                    tdf = pd.read_csv(ind_ds.file_path); xtc = tc if tc in tdf.columns else tdf.columns[-1]
                    xt = tdf.drop(columns=[xtc]); yt = tdf[xtc]; yte = le.transform(yt.astype(str))
                    ind_acc = m.score(xt, yte)
                    logger.info(f'    Ind: {ind_acc:.4f}  gap: {train_acc - ind_acc:.4f}')
                import pickle
                eid = f'retrain_{ds_id}_{algo}_{int(time.time())}'
                ed = os.path.join('experiments', eid); os.makedirs(ed, exist_ok=True)
                mp = os.path.join(ed, 'model.pkl')
                with open(mp, 'wb') as f: pickle.dump({'model': m, 'labels': le.classes_.tolist(), 'features': list(X.columns)}, f)
                mr = ModelRecord(name=f'{ds.name} - {algo} (anti-overfit)', model_type='classification', framework='sklearn', model_file_path=mp, hyperparameters_json=json.dumps(params, ensure_ascii=False), accuracy=float(train_acc), f1_score=float(train_acc), training_dataset_id=ds.id, status='trained', is_public=False, owner_id=admin.id if admin else 1)
                mr.set_metrics({'cv_mean': float(train_acc), 'cv_std': float(cvs.std()), 'cv_folds': 5, 'train_accuracy': float(m.score(X, ye))})
                db.session.add(mr); db.session.commit()
                if ind_ds and ind_acc is not None:
                    from app.services.inference_service import ModelInferenceService
                    ir = ModelInferenceService.test_model_with_split(mr, test_dataset=ind_ds)
                    if ir.get('success'):
                        mr.set_independent_metrics({'ind_test_accuracy': ir.get('accuracy'), 'ind_test_f1_macro': ir.get('f1_macro'), 'test_dataset_name': ind_ds.name, 'test_dataset_uuid': ind_ds.uuid, 'collection_method': ind_ds.collection_method})
                        mr.independent_test_dataset_id = ind_ds.id; db.session.commit()
                logger.info(f'    [OK] {mr.uuid}')
                results.append({'algo': algo, 'status': 'ok', 'uuid': mr.uuid, 'cv': float(train_acc), 'ind': float(ind_acc) if ind_acc else None})
            except Exception as e:
                logger.error(f'    [FAIL] {e}'); db.session.rollback()
                results.append({'algo': algo, 'status': 'failed', 'error': str(e)})
        return results

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--dry-run', action='store_true'); a = p.parse_args()
    all_r = {}
    for did in DATASET_IDS: all_r[did] = retrain(did, a.dry_run)
    ok = sum(1 for rl in all_r.values() for r in rl if r['status']=='ok')
    logger.info(f'Done. OK={ok}')
    for did, rl in all_r.items():
        for r in rl:
            if r['status']=='ok': logger.info(f'  ds={did} {r["algo"]}: cv={r["cv"]:.4f} ind={r.get("ind",0):.4f}')
