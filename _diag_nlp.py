import sys, pickle, os
sys.path.insert(0, '.')
from app import create_app, db
from app.models.model_record import ModelRecord

app = create_app()
with app.app_context():
    model = db.session.get(ModelRecord, 856)
    if model and model.model_file_path and os.path.exists(model.model_file_path):
        with open(model.model_file_path, 'rb') as f:
            bundle = pickle.load(f)
        print('=== RF Model (id=856) ===')
        print('Bundle keys:', list(bundle.keys()))
        if 'vectorizer' in bundle:
            print('vectorizer:', type(bundle['vectorizer']).__name__)
            print('vocab size:', len(bundle['vectorizer'].vocabulary_))
        if 'model' in bundle and 'vectorizer' in bundle:
            m = bundle['model']
            v = bundle['vectorizer']
            print('model type:', type(m).__name__)
            if hasattr(m, 'n_estimators'):
                print('n_estimators:', m.n_estimators)
            if hasattr(m, 'max_depth'):
                print('max_depth:', m.max_depth)
            if hasattr(m, 'min_samples_leaf'):
                print('min_samples_leaf:', m.min_samples_leaf)
            try:
                test_vec = v.transform(['太好看了，满分满分满分'])
                pred = m.predict(test_vec)
                proba = m.predict_proba(test_vec)
                labels = bundle.get('class_labels', ['负面','正面'])
                print('prediction:', pred, '->', labels[int(pred[0])] if labels else pred)
                print('probabilities:', proba)
            except Exception as e:
                print('predict error:', e)
        if 'feature_names' in bundle:
            print('features (first 5):', bundle['feature_names'][:5])

    print()
    model2 = db.session.get(ModelRecord, 858)
    if model2 and model2.model_file_path and os.path.exists(model2.model_file_path):
        with open(model2.model_file_path, 'rb') as f:
            bundle2 = pickle.load(f)
        print('=== LogisticReg Model (id=858) ===')
        print('Bundle keys:', list(bundle2.keys()))
        if 'vectorizer' in bundle2:
            v = bundle2['vectorizer']
            try:
                test = v.transform(['太好看了，满分满分满分'])
                print('Test transform shape:', test.shape)
                print('nnz:', test.nnz)
            except Exception as e:
                print('transform error:', e)
        if 'model' in bundle2:
            m2 = bundle2['model']
            print('model type:', type(m2).__name__)
            try:
                test_vec = bundle2['vectorizer'].transform(['太好看了，满分满分满分'])
                pred = m2.predict(test_vec)
                proba = m2.predict_proba(test_vec)
                print('prediction:', pred)
                print('probabilities:', proba)
            except Exception as e:
                print('predict error:', e)

    print()
    model3 = db.session.get(ModelRecord, 859)
    if model3 and model3.model_file_path and os.path.exists(model3.model_file_path):
        with open(model3.model_file_path, 'rb') as f:
            bundle3 = pickle.load(f)
        print('=== SVM Model (id=859) ===')
        print('Bundle keys:', list(bundle3.keys()))
        if 'model' in bundle3 and 'vectorizer' in bundle3:
            m3 = bundle3['model']
            v3 = bundle3['vectorizer']
            try:
                test_vec = v3.transform(['太好看了，满分满分满分'])
                pred = m3.predict(test_vec)
                proba = m3.predict_proba(test_vec)
                print('prediction:', pred)
                print('probabilities:', proba)
            except Exception as e:
                print('predict error:', e)
