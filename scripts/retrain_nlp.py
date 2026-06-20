"""
============================================
NLP 模型重新训练脚本 (Batch B)
============================================
使用真实中文文本 + TfidfVectorizer 管道
每个模型自动保存 vectorizer + class_labels

用法: python scripts/retrain_nlp.py [OPTIONS]
  --quick              只训练1个模型快速验证
  --dataset SUBSTR     按数据集名称筛选 (大小写不敏感), 如: --dataset chnsenticorp
  --algo ALGOS         指定算法: all (全部6个) 或逗号分隔, 如: --algo random_forest,svm
  --max-features N     TF-IDF 最大特征数 (默认: 2000)
  --min-df N           最小文档频率 (默认: 2)
  --max-df F           最大文档频率比例 (默认: 0.9)
  --balance MODE       重采样: smote (默认) / undersample / none
  --cv N               交叉验证折数 (默认: 5, 0=禁用)
  --augment N          数据增强倍数 (默认: 0=禁用, 建议2)
  --workers N          并行训练worker数 (默认: 2)
  --no-verify          跳过测试句子验证 (默认会验证10句)
  --output-json PATH   输出JSON报告到文件
"""
import os
import sys
import json
import time
import logging
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.user import User
from app.models.dataset import Dataset
from app.models.training_job import TrainingJob
from app.models.model_record import ModelRecord
from app.services.training_service import TrainingService
from app.executor.trainers.sklearn_trainer import SklearnTrainer
from app.executor.callbacks import TrainingCallback
from app.executor.trainers.pytorch_trainer import PyTorchTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / 'uploads' / 'datasets'

# ===================================================================
# NLP 数据集注册 (text,label CSV)
# ===================================================================
NLP_DATASETS = [
    {'csv': 'chnsenticorp_hotel.csv',
     'name': 'ChnSentiCorp-Hotel-Reviews',
     'desc': 'ChnSentiCorp 真实酒店评论情感数据'},
    {'csv': 'douban_reviews.csv',
     'name': 'Douban-Movie-Reviews',
     'desc': '豆瓣电影评论情感数据'},
    {'csv': 'shopping_reviews.csv',
     'name': 'Shopping-Reviews',
     'desc': '电商购物评论情感数据'},
]

# ===================================================================
# sklearn 分类算法 (全部 6 个)
# ===================================================================
ALL_SKLEARN_ALGOS = [
    {'algo': 'random_forest',       'cn': 'RandomForest'},
    {'algo': 'logistic_regression', 'cn': 'LogisticReg'},
    {'algo': 'svm',                 'cn': 'SVM'},
    {'algo': 'knn',                 'cn': 'KNN'},
    {'algo': 'gradient_boosting',   'cn': 'GradientBoost'},
    {'algo': 'decision_tree',       'cn': 'DecisionTree'},
]

# 默认算法集 (向后兼容: 经典3算法)
ALGORITHMS = ALL_SKLEARN_ALGOS[:3]

# ===================================================================
# 数据集特定测试句子 (5正+5负=10句, 覆盖多样表达)
# ===================================================================
DATASET_TEST_SENTENCES = {
    'chnsenticorp': [
        # --- 正面 (5句) ---
        ("不错，在同等档次酒店中应该是值得推荐的！", "正面"),
        ("商务大床房，房间很大，整体感觉经济实惠不错!", "正面"),
        ("酒店位置很好，交通方便，服务态度也不错", "正面"),
        ("环境整洁干净，早餐种类丰富，下次还来", "正面"),
        ("性价比挺高的，虽然不豪华但住着舒服", "正面"),
        # --- 负面 (5句) ---
        ("早餐太差，无论去多少人，那边也不加食品的。", "负面"),
        ("宾馆在小街道上，不大好找，房间很小，确实挺小。", "负面"),
        ("隔音太差了，隔壁说话都能听见，一晚没睡好", "负面"),
        ("前台服务态度冷漠，房间有异味，很失望", "负面"),
        ("图片和实际完全不符，设施老旧，不值这个价", "负面"),
    ],
    'douban': [
        # --- 正面 (5句) ---
        ("这部电影真的很好看，剧情精彩，演员演技在线，强烈推荐！", "正面"),
        ("不错的电影，导演功力深厚，画面精美，值得一看。", "正面"),
        ("配乐和摄影都很出色，故事虽然简单但很打动人", "正面"),
        ("笑中带泪，节奏把控得很好，今年最佳国产片", "正面"),
        ("虽然小众但品质很高，推荐给喜欢文艺片的朋友", "正面"),
        # --- 负面 (5句) ---
        ("烂片，剧情混乱，演技尴尬，浪费时间和金钱，不推荐。", "负面"),
        ("太差了，根本看不下去，中途就退场了，千万别看。", "负面"),
        ("跟原著完全没法比，改编得一塌糊涂，太失望了", "负面"),
        ("全程尿点，特效五毛，编剧应该被拉黑", "负面"),
        ("被营销骗了，完全不值票价，后悔看了这部", "负面"),
    ],
    'shopping': [
        # --- 正面 (5句) ---
        ("质量很好，做工精细，卖家发货快，值得购买，好评！", "正面"),
        ("性价比很高，用了一段时间了，没什么问题，满意。", "正面"),
        ("包装很用心，客服态度好，物流也很快", "正面"),
        ("买了三个多月了，质量依然很好，耐用", "正面"),
        ("跟描述的一致，颜色也好看，会回购的", "正面"),
        # --- 负面 (5句) ---
        ("质量太差，用了一次就坏了，卖家还不理人，差评。", "负面"),
        ("跟描述完全不符，做工粗糙，退货还要自己出运费，上当。", "负面"),
        ("味道很大，晾了好几天还是有刺鼻的气味，不敢用", "负面"),
        ("尺码严重偏小，客服推卸责任，购物体验极差", "负面"),
        ("明显是假货，细节和正品差太多了，必须差评", "负面"),
    ],
}

# 根据数据集 CSV 名获取测试句子
def get_test_sentences(dataset_name: str):
    """根据数据集名称匹配测试句子，fallback 到 chnsenticorp"""
    name_lower = (dataset_name or '').lower()
    for key in DATASET_TEST_SENTENCES:
        if key in name_lower:
            return DATASET_TEST_SENTENCES[key]
    return DATASET_TEST_SENTENCES['chnsenticorp']

# 支持 class_weight 的算法 (解决类别不平衡)
_ALGOS_WITH_CLASS_WEIGHT = {'random_forest', 'logistic_regression', 'svm', 'decision_tree'}


def ensure_datasets(app, admin):
    """确保 NLP 数据集已注册到数据库"""
    datasets = []
    with app.app_context():
        import pandas as pd
        for cfg in NLP_DATASETS:
            csv_path = DATASETS_DIR / cfg['csv']
            if not csv_path.exists():
                logger.warning(f'跳过: {cfg["csv"]} (文件不存在)')
                continue

            existing = Dataset.query.filter_by(name=cfg['name']).first()
            if existing:
                logger.info(f'数据集已存在: {cfg["name"]} (id={existing.id})')
                datasets.append(existing)
                continue

            df = pd.read_csv(csv_path)
            ds = Dataset(
                name=cfg['name'],
                owner_id=admin.id,
                file_path=str(csv_path),
                file_format='csv',
                file_size=csv_path.stat().st_size,
                row_count=len(df),
                column_count=len(df.columns),
                category='nlp',
                status='ready',
                is_public=True,
                summary_json=json.dumps({
                    'columns': list(df.columns),
                    'target_column': 'label',
                    'label_distribution': {
                        str(k): int(v)
                        for k, v in df['label'].value_counts().items()
                    },
                }, ensure_ascii=False),
            )
            db.session.add(ds)
            db.session.commit()
            logger.info(f'数据集已注册: {cfg["name"]} ({len(df)} 行, id={ds.id})')
            datasets.append(ds)

    return datasets


def verify_model_predictions(model_path, test_sentences):
    """验证已训练模型的预测能力
    返回 (vec_ok, accuracy, test_results) 元组
    test_results: [(sentence, expected, predicted, correct), ...]
    """
    vec_ok = False
    accuracy = 0.0
    test_results = []

    if not model_path or not os.path.exists(model_path):
        return vec_ok, accuracy, test_results

    try:
        vectorizer = None
        clf = None
        class_labels = []

        # ── 加载模型 bundle ──
        if model_path.endswith('.pt'):
            config_path = model_path.replace('.pt', '_config.pkl')
            if not os.path.exists(config_path):
                return vec_ok, accuracy, test_results
            with open(config_path, 'rb') as f:
                cfg = pickle.load(f)
            vectorizer = cfg.get('vectorizer')
            class_labels = cfg.get('class_labels', [])
        else:
            with open(model_path, 'rb') as f:
                bundle = pickle.load(f)
            vectorizer = bundle.get('vectorizer')
            class_labels = bundle.get('class_labels', [])
            clf = bundle.get('model')

        vec_ok = vectorizer is not None

        # ── 运行测试句子 ──
        if vec_ok and clf is not None and test_sentences:
            for sentence, expected in test_sentences:
                try:
                    X_vec = vectorizer.transform([sentence])
                    if hasattr(X_vec, 'toarray'):
                        X_vec = X_vec.toarray()
                    pred_raw = clf.predict(X_vec)[0]

                    # ── 将预测值映射为可读标签 ──
                    # NOTE: sklearn 训练时 LabelEncoder 已将标签编码为 int,
                    #       因此 clf.classes_ = [0, 1] 而非 ['正面', '负面']
                    #       真实标签字符串存储在 bundle['class_labels'] 中
                    if class_labels and len(class_labels) > 0:
                        try:
                            idx = int(pred_raw)
                            pred_label = (str(class_labels[idx])
                                          if 0 <= idx < len(class_labels)
                                          else str(pred_raw))
                        except (ValueError, TypeError):
                            pred_label = str(pred_raw)
                    elif hasattr(clf, 'classes_') and clf.classes_ is not None:
                        cl = list(clf.classes_)
                        try:
                            idx = int(pred_raw)
                            pred_label = str(cl[idx]) if 0 <= idx < len(cl) else str(pred_raw)
                        except (ValueError, TypeError):
                            pred_label = str(pred_raw)
                    else:
                        pred_label = str(pred_raw)

                    correct = (pred_label == expected)
                    test_results.append((sentence, expected, pred_label, correct))
                except Exception as e:
                    test_results.append((sentence, expected, f'ERROR: {e}', False))

        if test_results:
            accuracy = sum(1 for _, _, _, c in test_results if c) / len(test_results)

        return vec_ok, accuracy, test_results

    except Exception as e:
        logger.warning(f'验证模型预测失败: {e}')
        return vec_ok, accuracy, test_results


def train_one(app, admin_id, dataset_id, algo_conf, idx,
              verify=True,
              nlp_max_features=2000,
              nlp_min_df=2,
              nlp_max_df=0.9,
              balance_mode='smote',
              cv_folds=5,
              augment_factor=0):
    """训练一个NLP模型"""
    algo = algo_conf['algo']
    cn = algo_conf['cn']
    use_pytorch = algo.startswith('mlp')

    with app.app_context():
        try:
            framework = 'pytorch' if use_pytorch else 'sklearn'
            admin = db.session.get(User, admin_id)
            dataset = db.session.get(Dataset, dataset_id)
            if not dataset:
                logger.error(f'[{idx}] 数据集不存在: id={dataset_id}')
                return None

            job, error = TrainingService.create_job(
                user=admin,
                name=f'NLP-{cn}-{dataset.name[:15]}',
                dataset_id=dataset.id,
                framework=framework,
                total_epochs=10 if use_pytorch else 5,
                ml_task_type='classification',
                algorithm=algo,
                target_column='label',
                test_size=0.2,
                model_type='nlp',
            )
            if error:
                logger.error(f'[{idx}] 创建任务失败: {error}')
                return None

            job = db.session.get(TrainingJob, job.id)
            model = db.session.get(ModelRecord, job.model_id)
            dataset = db.session.get(Dataset, dataset.id)

            logger.info(f'[{idx}] 开始: {model.name} ({algo})')

            # ── 基础超参数 ──
            hp = {
                'algorithm': algo,
                'task_type': 'classification',
                'target_column': 'label',
                'test_size': 0.2,
                'random_state': 42,
                'total_epochs': 10 if use_pytorch else 5,
                'hidden_layers': [128, 64] if use_pytorch else None,
                'nlp_max_features': nlp_max_features,
                'nlp_min_df': nlp_min_df,
                'nlp_max_df': nlp_max_df,
                'balance': balance_mode,
                'cv_folds': cv_folds,
                'augment_factor': augment_factor,
            }

            # ── 算法特定参数 ──
            _algo_params = {}
            if algo in _ALGOS_WITH_CLASS_WEIGHT:
                _algo_params['class_weight'] = 'balanced'

            # 各算法最佳实践参数 (Batch B 优化)
            if algo == 'logistic_regression':
                _algo_params['max_iter'] = 2000
                _algo_params['solver'] = 'liblinear'
            elif algo == 'svm':
                _algo_params['probability'] = True
                _algo_params['kernel'] = 'rbf'
            elif algo == 'knn':
                _algo_params['weights'] = 'distance'
                _algo_params['n_neighbors'] = 5
            elif algo == 'gradient_boosting':
                _algo_params['n_estimators'] = 200
                _algo_params['max_depth'] = 5
                _algo_params['min_samples_leaf'] = 10
            elif algo == 'random_forest':
                _algo_params['n_estimators'] = 200
                _algo_params['max_depth'] = 15
                _algo_params['min_samples_leaf'] = 5
            elif algo == 'decision_tree':
                _algo_params['max_depth'] = 10
                _algo_params['min_samples_leaf'] = 5

            if _algo_params:
                hp['algorithm_params'] = _algo_params

            hp = {k: v for k, v in hp.items() if v is not None}

            TrainerClass = PyTorchTrainer if use_pytorch else SklearnTrainer
            trainer = TrainerClass(job, dataset, hp)
            # NOTE: do NOT overwrite trainer.callback — BaseTrainer.__init__ already
            # creates a correct TrainingCallback(job.id). Overwriting with
            # TrainingCallback(job) (ORM object instead of int) breaks on_log().

            t0 = time.time()
            trainer.run()
            elapsed = time.time() - t0

            model = db.session.get(ModelRecord, model.id)
            accuracy = model.accuracy or 0

            # 验证 vectorizer
            vec_ok = False
            labels_ok = False
            model_path = model.model_file_path
            if model_path and os.path.exists(model_path):
                actual_path = model_path
                if model_path.endswith('.pt'):
                    config_path = model_path.replace('.pt', '_config.pkl')
                    if os.path.exists(config_path):
                        with open(config_path, 'rb') as f:
                            cfg = pickle.load(f)
                        vec_ok = 'vectorizer' in cfg
                        labels_ok = bool(cfg.get('class_labels'))
                else:
                    with open(model_path, 'rb') as f:
                        bundle = pickle.load(f)
                    vec_ok = bundle.get('vectorizer') is not None
                    labels_ok = bool(bundle.get('class_labels'))

            logger.info(
                f'[{idx}] DONE: acc={accuracy:.3f}, time={elapsed:.0f}s, '
                f'vectorizer={"YES" if vec_ok else "NO"}, '
                f'class_labels={"YES" if labels_ok else "NO"}'
            )

            # ── 测试句子验证 ──
            if verify and vec_ok:
                ds_tests = get_test_sentences(dataset.name)
                _, test_acc, test_results = verify_model_predictions(
                    model.model_file_path, ds_tests
                )
                if test_results:
                    correct = sum(1 for _, _, _, c in test_results if c)
                    total = len(test_results)
                    status = 'PASS' if test_acc >= 0.75 else 'WARN'
                    logger.info(
                        f'[{idx}] TEST-SENTENCES: {correct}/{total} correct '
                        f'(acc={test_acc:.2%}), status={status}'
                    )
                    for sent, exp, pred, ok in test_results:
                        mark = 'OK' if ok else 'X'
                        logger.info(f'[{idx}]   [{mark}] "{sent[:40]}..." -> {pred} (exp: {exp})')

            return model

        except Exception as e:
            logger.error(f'[{idx}] 失败: {e}')
            import traceback
            traceback.print_exc()
            return None


def parse_args():
    """使用 argparse 解析命令行参数 (替代手动 sys.argv 循环)。"""
    import argparse

    parser = argparse.ArgumentParser(
        description='NLP 模型重新训练脚本 — 使用真实中文文本 + TfidfVectorizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python scripts/retrain_nlp.py --quick                           # 快速验证1个模型
  python scripts/retrain_nlp.py --dataset chnsenticorp            # 仅训练指定数据集
  python scripts/retrain_nlp.py --algo all --workers 4            # 全部算法, 4 worker
  python scripts/retrain_nlp.py --balance undersample --augment 2 # 欠采样 + 数据增强
        ''',
    )
    parser.add_argument('--quick', action='store_true',
                        help='只训练1个模型快速验证')
    parser.add_argument('--no-verify', action='store_true',
                        help='跳过测试句子验证 (默认会验证10句)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='按数据集名称筛选 (大小写不敏感), 如: chnsenticorp')
    parser.add_argument('--algo', type=str, default=None,
                        help='指定算法: all (全部6个) 或逗号分隔, 如: svm,random_forest')
    parser.add_argument('--max-features', type=int, default=2000,
                        help='TF-IDF 最大特征数 (默认: 2000)')
    parser.add_argument('--min-df', type=int, default=2,
                        help='最小文档频率 (默认: 2)')
    parser.add_argument('--max-df', type=float, default=0.9,
                        help='最大文档频率比例 (默认: 0.9)')
    parser.add_argument('--balance', type=str, default='smote',
                        choices=['smote', 'undersample', 'none'],
                        help='重采样策略 (默认: smote)')
    parser.add_argument('--cv', dest='cv_folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5, 0=禁用)')
    parser.add_argument('--augment', dest='augment_factor', type=int, default=0,
                        help='数据增强倍数 (默认: 0=禁用, 建议2)')
    parser.add_argument('--workers', type=int, default=2,
                        help='并行训练worker数 (默认: 2)')
    parser.add_argument('--output-json', type=str, default=None,
                        help='输出JSON报告到文件')

    return parser.parse_args()


def main():
    args = parse_args()

    quick = args.quick
    verify_flag = not args.no_verify

    dataset_filter = args.dataset.lower() if args.dataset else None

    # ── 解析 --algo: None=默认3算法, 'all'=全部6个, list=指定列表 ──
    algo_filter = None
    if args.algo:
        if args.algo == 'all':
            algo_filter = 'all'
        else:
            algo_filter = [a.strip() for a in args.algo.split(',') if a.strip()]

    nlp_max_features = args.max_features
    nlp_min_df = args.min_df
    nlp_max_df = args.max_df

    balance_mode = args.balance if args.balance != 'none' else None

    cv_folds = args.cv_folds
    if cv_folds < 0:
        cv_folds = 0

    augment_factor = args.augment_factor
    if augment_factor < 2:
        augment_factor = 0

    workers = args.workers
    if workers < 1:
        workers = 1

    output_json = args.output_json

    app = create_app()

    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            logger.error('管理员用户不存在, 请先运行 seed_data.py')
            return
        logger.info(f'用户: {admin.username}')

    # Phase 1: 注册数据集
    logger.info('=== Phase 1: 注册 NLP 数据集 ===')
    datasets = ensure_datasets(app, admin)
    if not datasets:
        logger.error('没有可用的 NLP 数据集')
        return

    # ── 按名称筛选数据集 ──
    if dataset_filter:
        filtered = [ds for ds in datasets if dataset_filter in ds.name.lower()]
        if not filtered:
            logger.error(
                f'没有匹配 "{dataset_filter}" 的数据集. '
                f'可用: {[ds.name for ds in datasets]}'
            )
            return
        logger.info(
            f'数据集筛选 "{dataset_filter}": '
            f'{len(filtered)}/{len(datasets)} 个匹配'
        )
        datasets = filtered

    # ── 确定算法集 ──
    if algo_filter == 'all':
        algos = ALL_SKLEARN_ALGOS
    elif isinstance(algo_filter, list):
        # 按名称匹配算法
        matched = []
        for name in algo_filter:
            found = None
            for a in ALL_SKLEARN_ALGOS:
                if a['algo'] == name or a['cn'].lower() == name.lower():
                    found = a
                    break
            if found:
                matched.append(found)
            else:
                logger.warning(
                    f'未知算法 "{name}", 可用: '
                    f'{[a["algo"] for a in ALL_SKLEARN_ALGOS]}'
                )
        algos = matched if matched else ALGORITHMS
    else:
        algos = ALGORITHMS
    logger.info(f'算法集 ({len(algos)}): {[a["cn"] for a in algos]}')

    # Phase 2: 训练模型
    models_per_ds = len(algos)
    total_expected = len(datasets) * models_per_ds
    logger.info(
        f'=== Phase 2: 训练 NLP 模型 '
        f'({len(datasets)} ds x {models_per_ds} algo = {total_expected} 模型) ==='
    )

    if quick:
        logger.info('QUICK 模式: 仅训练 1 个模型')
        datasets = datasets[:1]
        algos = algos[:1]
        models_per_ds = 1

    total = 0
    success = 0
    results = []  # 收集所有训练结果用于 JSON 报告

    admin_id = admin.id
    for ds in datasets:
        for algo_conf in algos:
            total += 1
            # 自动对小数据集启用增强 (Douban 800行/Shopping 700行)
            ds_augment = augment_factor
            if augment_factor <= 0 and ds.row_count and ds.row_count < 1000:
                ds_augment = 2  # 自动2x增强
                logger.info(f'自动启用数据增强: {ds.name} ({ds.row_count}行) -> factor=2')

            model = train_one(
                app, admin_id, ds.id, algo_conf, total,
                verify=verify_flag,
                nlp_max_features=nlp_max_features,
                nlp_min_df=nlp_min_df,
                nlp_max_df=nlp_max_df,
                balance_mode=balance_mode,
                cv_folds=cv_folds if cv_folds > 0 else 0,
                augment_factor=ds_augment,
            )
            result = {
                'index': total,
                'dataset': ds.name,
                'algorithm': algo_conf['algo'],
                'algorithm_cn': algo_conf['cn'],
                'status': 'failed',
                'accuracy': None,
                'duration_seconds': None,
                'error': None,
            }
            if model and model.status == 'trained':
                success += 1
                result['status'] = 'trained'
                result['accuracy'] = model.accuracy
                result['duration_seconds'] = model.training_duration_seconds
                result['model_uuid'] = model.uuid
                result['model_name'] = model.name
            elif model:
                result['error'] = f'status={model.status}'
            results.append(result)

    # 汇总
    logger.info(f'\n=== NLP 模型训练完成: {success}/{total} ===')

    # 写入 JSON 报告
    if output_json:
        import datetime as _dt
        report = {
            'title': 'NLP模型重训练报告',
            'generated_at': _dt.datetime.now().isoformat(),
            'config': {
                'algorithms': [a['algo'] for a in algos],
                'datasets': [ds.name for ds in datasets],
                'cv_folds': cv_folds,
                'augment_factor': augment_factor,
                'balance_mode': balance_mode,
                'max_features': nlp_max_features,
                'min_df': nlp_min_df,
                'max_df': nlp_max_df,
            },
            'summary': {
                'total': total,
                'success': success,
                'failed': total - success,
            },
            'results': results,
        }
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f'JSON 报告已写入: {output_json}')

    # Phase 3: 验证所有 NLP 模型
    logger.info('=== Phase 3: 验证模型文件 ===')
    with app.app_context():
        nlp_models = ModelRecord.query.filter_by(
            model_type='nlp', status='trained'
        ).order_by(ModelRecord.created_at.desc()).limit(20).all()
        for m in nlp_models:
            path = m.model_file_path
            if path and os.path.exists(path):
                try:
                    actual = path
                    v = None
                    cl = []
                    if path.endswith('.pt'):
                        cp = path.replace('.pt', '_config.pkl')
                        if os.path.exists(cp):
                            with open(cp, 'rb') as f:
                                b = pickle.load(f)
                            v = b.get('vectorizer')
                            cl = b.get('class_labels', [])
                    elif path.endswith('.pkl'):
                        with open(path, 'rb') as f:
                            b = pickle.load(f)
                        v = b.get('vectorizer')
                        cl = b.get('class_labels', [])
                    logger.info(f'  {m.name}: acc={m.accuracy}, vec={"Y" if v else "N"}, labels={cl}')
                except Exception as e:
                    logger.info(f'  {m.name}: acc={m.accuracy}, (load err: {e})')
            else:
                logger.info(f'  {m.name}: file not found')


if __name__ == '__main__':
    main()
