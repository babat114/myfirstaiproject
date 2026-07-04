"""
数据集审计 — 全量分类为 KEEP/DELETE/UNCERTAIN

扫描所有数据集及其关联模型，基于来源、文件名模式、DATASET_MANIFEST
白名单等规则精确分类。

用法:
    python scripts/audit_datasets.py                  # 输出审计报告
    python scripts/audit_datasets.py --output-json report.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.dataset import Dataset
from app.models.model_record import ModelRecord

# ── 真实数据集白名单 ──
# 1. real_*.csv 中 source != 'synth' 的 (来自 fetch_real_datasets.py DATASET_MANIFEST)
REAL_MANIFEST_KEYS = {
    # OpenML 来源
    'adult', 'auto_mpg', 'bank_marketing', 'bodyfat', 'boston_housing',
    'car_eval', 'concrete', 'cpu_performance', 'credit_approval',
    'ecoli', 'energy_efficiency', 'german_credit', 'glass',
    'mnist_small', 'mushroom', 'pima_diabetes', 'seeds',
    'spambase', 'student_performance', 'telco_churn', 'titanic',
    'vehicle', 'wine_quality_binary', 'wine_quality_red',
    # sklearn 内置
    'california_housing', 'blobs_varied', 'moons',
}
# 2. real_*.csv 中 source=synth 或仅 synth 条目的数据集 (需要标记为合成)
SYNTH_MANIFEST_KEYS = {
    'air_quality', 'circles', 'diabetes_regression', 'mall_customers',
    'moons_dense',
}

# 3. NLP 中文真实数据集
REAL_NLP_NAMES = {
    'ChnSentiCorp-Hotel', 'Douban-Movie', 'Shopping-Review',
}

# 4. sklearn 真实医学数据
REAL_SKLEARN_NAMES = {
    'Breast-Cancer-Diagnosis-Biology',
}

# ── 合成数据集文件名模式 ──
SYNTH_PATTERNS = [
    # GEN80: {type}_NN_{rows}rows.csv
    r'^(classification|regression|clustering|nlp|computer_vision|generative|reinforcement|other)_\d+_\d+rows\.csv$',
    # GEN_MULTITYPE
    r'^(Binary_Clean_20K|Binary_Linear_15K|Multiclass3_20K|Multiclass5_25K|Binary_Clustered_18K)\.csv$',
    # GEN_MULTITYPE_NAMED
    r'^(Regression-Large-50K|Clustering-Blobs-40K|NLP-TextFeatures-30K|CV-ImageFeatures-35K|RL-StateAction-45K|Gen-LatentSpace-60K|Other-AnomalyMix-55K)',
    # PUBLIC_SYNTH_DL
    r'^(DL-Class-50K|DL-Reg-30K)\.csv$',
    # CATEGORY_GEN_SYNTH
    r'^(Stock-Price-TimeSeries|Credit-Risk-Finance)\.csv$',
    # SYNTH_NAMED
    r'^Synthetic-Anomaly-Detection\.csv$',
    # UUID_UPLOAD (copies) — UUID with dashes
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.csv$',
    # UUID_UPLOAD (copies) — 32-char hex without dashes
    r'^[0-9a-f]{32}\.csv$',
    # PUBLIC_SYNTH_DL (actual filenames)
    r'^public_synthetic_dl_(large|regression)_\d{8}_\d{6}\.csv$',
    # CATEGORY_GEN (actual filenames)
    r'^(stock_price_timeseries|credit_risk_finance|synthetic_anomaly_detection)\.csv$',
    # GEN_MULTITYPE_NAMED with timestamp suffix
    r'^(Regression-Large-50K|Clustering-Blobs-40K|NLP-TextFeatures-30K|CV-ImageFeatures-35K|RL-StateAction-45K|Gen-LatentSpace-60K|Other-AnomalyMix-55K)_\d{8}_\d{6}',
    # GEN80 by name pattern (Chinese bracket names from generate_80_datasets.py)
    r'^generative_o\d_\d+rows$',
    # classification_30k leftover
    r'^classification_30k\.csv$',
]

import re
_SYNTH_REGEXES = [re.compile(p) for p in SYNTH_PATTERNS]


def _get_filename(path: str) -> str:
    """Extract basename from file path."""
    if not path:
        return ''
    return os.path.basename(path)


def _matches_synth_pattern(filename: str) -> bool:
    """Check if filename matches any known synthetic pattern."""
    for regex in _SYNTH_REGEXES:
        if regex.match(filename):
            return True
    return False


def _extract_real_key(filename: str) -> str:
    """Extract the manifest key from a real_*.csv filename."""
    if filename.startswith('real_') and filename.endswith('.csv'):
        return filename.replace('real_', '').replace('.csv', '')
    return ''


def _classify_dataset(ds: Dataset) -> dict:
    """Classify a single dataset as KEEP, DELETE_SYNTH, DELETE_SYNTH_WITH_MODELS, or UNCERTAIN."""
    fname = _get_filename(ds.file_path or '')
    name = ds.name or ''

    # Count associated models
    model_count = ModelRecord.query.filter_by(training_dataset_id=ds.id).count()

    result = {
        'dataset_id': ds.id,
        'name': name,
        'uuid': ds.uuid,
        'file_path': ds.file_path,
        'filename': fname,
        'category': ds.category,
        'row_count': ds.row_count,
        'column_count': ds.column_count,
        'status': ds.status,
        'is_public': ds.is_public,
        'model_count': model_count,
        'classification': '',
        'reason': '',
    }

    # Rule 1: real_*.csv → check manifest
    if fname.startswith('real_') and fname.endswith('.csv'):
        key = _extract_real_key(fname)
        if key in REAL_MANIFEST_KEYS:
            result['classification'] = 'KEEP'
            result['reason'] = f'真实数据集 (OpenML/sklearn): {key}'
        elif key in SYNTH_MANIFEST_KEYS:
            if model_count > 0:
                result['classification'] = 'DELETE_SYNTH_WITH_MODELS'
                result['reason'] = f'real_合成兜底数据集 (source=synth): {key}'
            else:
                result['classification'] = 'DELETE_SYNTH'
                result['reason'] = f'real_合成兜底数据集 (source=synth): {key}'
        else:
            # Not in manifest → check if it's a known deprecated real_ dataset
            result['classification'] = 'UNCERTAIN'
            result['reason'] = f'real_*.csv 但不在manifest中: {key}'
        return result

    # Rule 2: NLP Chinese real datasets
    for nlp_name in REAL_NLP_NAMES:
        if nlp_name.lower() in name.lower():
            result['classification'] = 'KEEP'
            result['reason'] = f'NLP中文真实数据集: {nlp_name}'
            return result

    # Rule 3: sklearn real medical
    for sk_name in REAL_SKLEARN_NAMES:
        if sk_name.lower() in name.lower():
            result['classification'] = 'KEEP'
            result['reason'] = f'sklearn真实医学数据: {sk_name}'
            return result

    # Rule 4: Known synthetic patterns from filename
    if _matches_synth_pattern(fname):
        if model_count > 0:
            result['classification'] = 'DELETE_SYNTH_WITH_MODELS'
            result['reason'] = f'合成数据集 (模式匹配): {fname}'
        else:
            result['classification'] = 'DELETE_SYNTH'
            result['reason'] = f'合成数据集 (模式匹配): {fname}'
        return result

    # Rule 5: User-upload with UUID/hex filename → likely copies of GEN80
    if fname and (re.match(r'^[0-9a-f]{8}-.*\.csv$', fname) or
                  re.match(r'^[0-9a-f]{32}\.csv$', fname)):
        if model_count > 0:
            result['classification'] = 'DELETE_SYNTH_WITH_MODELS'
            result['reason'] = f'UUID/hex上传副本 (疑似合成数据拷贝)'
        else:
            result['classification'] = 'DELETE_SYNTH'
            result['reason'] = f'UUID/hex上传副本 (疑似合成数据拷贝)'
        return result

    # Rule 6: GEN80 Chinese bracket names (generate_80_datasets.py pattern)
    GEN80_NAME_PREFIXES = [
        '[分类]', '[回归]', '[聚类]', '[自然语言处理]',
        '[计算机视觉]', '[强化学习]', '[生成]', '[其他]',
    ]
    for prefix in GEN80_NAME_PREFIXES:
        if prefix in name:
            if model_count > 0:
                result['classification'] = 'DELETE_SYNTH_WITH_MODELS'
                result['reason'] = f'GEN80合成数据集 (名称匹配: {prefix})'
            else:
                result['classification'] = 'DELETE_SYNTH'
                result['reason'] = f'GEN80合成数据集 (名称匹配: {prefix})'
            return result

    # Rule 7: GENERATIVE named (like generative_o4_20000rows)
    if re.match(r'^generative_o\d+_\d+rows$', fname.replace('.csv', '')):
        if model_count > 0:
            result['classification'] = 'DELETE_SYNTH_WITH_MODELS'
            result['reason'] = f'GENERATIVE合成数据集 (名称模式)'
        else:
            result['classification'] = 'DELETE_SYNTH'
            result['reason'] = f'GENERATIVE合成数据集 (名称模式)'
        return result

    # Rule 8: PUBLIC_SYNTH_DL by name
    if 'DL-Class' in name or 'DL-Reg' in name:
        result['classification'] = 'DELETE_SYNTH'
        result['reason'] = f'PUBLIC_SYNTH_DL合成数据集'
        return result

    # Fallback
    result['classification'] = 'UNCERTAIN'
    result['reason'] = f'无法自动分类: name={name[:60]}, file={fname}'
    return result


def run_audit(app, output_json: str = None):
    """Run full audit and generate report."""
    with app.app_context():
        all_datasets = Dataset.query.order_by(Dataset.id).all()

        results = []
        keep = []
        delete_synth = []
        delete_synth_with_models = []
        uncertain = []

        for ds in all_datasets:
            entry = _classify_dataset(ds)
            results.append(entry)

            if entry['classification'] == 'KEEP':
                keep.append(entry)
            elif entry['classification'] == 'DELETE_SYNTH':
                delete_synth.append(entry)
            elif entry['classification'] == 'DELETE_SYNTH_WITH_MODELS':
                delete_synth_with_models.append(entry)
            else:
                uncertain.append(entry)

        # ── Statistics ──
        total_models_to_delete = sum(
            d['model_count'] for d in delete_synth_with_models
        )
        total_kept_models = sum(d['model_count'] for d in keep)

        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_datasets': len(all_datasets),
                'KEEP': len(keep),
                'DELETE_SYNTH': len(delete_synth),
                'DELETE_SYNTH_WITH_MODELS': len(delete_synth_with_models),
                'UNCERTAIN': len(uncertain),
                'total_models_to_delete': total_models_to_delete,
                'total_kept_models': total_kept_models,
                'current_trained_models': ModelRecord.query.filter_by(status='trained').count(),
            },
            'KEEP': keep,
            'DELETE_SYNTH': delete_synth,
            'DELETE_SYNTH_WITH_MODELS': delete_synth_with_models,
            'UNCERTAIN': uncertain,
        }

        # ── Print Summary ──
        print(f"\n{'=' * 70}")
        print(f"  数据集审计报告")
        print(f"{'=' * 70}")
        print(f"  总计数据集:     {len(all_datasets):>5}")
        print(f"  保留 (KEEP):    {len(keep):>5}  ({total_kept_models} 模型)")
        print(f"  删除-纯合成:    {len(delete_synth):>5}  (0 模型)")
        print(f"  删除-合成+模型: {len(delete_synth_with_models):>5}  ({total_models_to_delete} 模型)")
        print(f"  不确定:         {len(uncertain):>5}")
        print(f"{'=' * 70}")
        print(f"  待删除数据集总计: {len(delete_synth) + len(delete_synth_with_models)}")
        print(f"  待删除模型总计:   {total_models_to_delete}")
        print(f"  删除后保留模型:   {total_kept_models}")

        # Print KEEP
        print(f"\n{'─' * 70}")
        print(f"  [KEEP] 保留 — {len(keep)} 个真实数据集:")
        for d in sorted(keep, key=lambda x: x['dataset_id']):
            print(f"    ID={d['dataset_id']:<4} models={d['model_count']:<4} "
                  f"{d['filename'][:40]:40s}  [{d['reason']}]")

        # Print DELETE_SYNTH_WITH_MODELS
        if delete_synth_with_models:
            print(f"\n{'─' * 70}")
            print(f"  [DELETE] 合成数据集(有模型) — {len(delete_synth_with_models)} 个:")
            for d in sorted(delete_synth_with_models, key=lambda x: -x['model_count']):
                print(f"    ID={d['dataset_id']:<4} models={d['model_count']:<4} "
                      f"{d['filename'][:40]:40s}  [{d['reason']}]")

        # Print DELETE_SYNTH (sample)
        if delete_synth:
            print(f"\n{'─' * 70}")
            print(f"  [DELETE] 纯合成数据集(无模型) — {len(delete_synth)} 个:")
            # Show first 15 + last 5
            for d in sorted(delete_synth, key=lambda x: x['dataset_id'])[:15]:
                print(f"    ID={d['dataset_id']:<4} {d['filename'][:50]:50s}")
            if len(delete_synth) > 20:
                print(f"    ... 省略 {len(delete_synth) - 20} 个 ...")
                for d in sorted(delete_synth, key=lambda x: x['dataset_id'])[-5:]:
                    print(f"    ID={d['dataset_id']:<4} {d['filename'][:50]:50s}")

        if uncertain:
            print(f"\n{'─' * 70}")
            print(f"  [UNCERTAIN] 无法自动分类 — {len(uncertain)} 个:")
            for d in uncertain:
                print(f"    ID={d['dataset_id']:<4} models={d['model_count']:<4} "
                      f"{d['filename'][:50]:50s}  [{d['reason']}]")

        # ── Save JSON ──
        if output_json is None:
            experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'
            experiments_dir.mkdir(exist_ok=True)
            output_json = str(experiments_dir / 'dataset_audit_report.json')

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n[Report] 审计报告已保存: {output_json}")

        return report


def main():
    parser = argparse.ArgumentParser(description='数据集审计 — 分类所有数据集')
    parser.add_argument('--output-json', help='输出JSON路径')
    args = parser.parse_args()

    app = create_app()
    run_audit(app, args.output_json)


if __name__ == '__main__':
    main()
