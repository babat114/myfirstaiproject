"""
============================================
数据集分类迁移脚本
将 category 字段从 ENUM 迁移到 VARCHAR(50)
并智能更新现有数据集的分类
============================================
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models.dataset import Dataset
import pandas as pd

# 智能分类推断 (与 dataset_service.py 中的逻辑一致)
_CATEGORY_KEYWORDS = [
    (['nlp', 'text', 'language', 'corpus', 'sentiment', 'word', 'document',
      '自然语言', '文本', '语料', '情感'], 'nlp'),
    (['vision', 'image', 'mnist', 'fashion', 'cifar', 'photo', 'picture',
      'pixel', '图像', '图片', '视觉', '手写', 'cv'], 'vision'),
    (['time_series', 'timeseries', 'temporal', 'stock', 'weather',
      '时序', '时间序列', '股票', '天气', 'sensor'], 'time_series'),
    (['regression', 'reg', 'price', 'housing', 'california',
      '回归', '房价'], 'regression'),
    (['cluster', 'clustering', 'blob', 'segment',
      '聚类', '分群'], 'clustering'),
    (['biology', 'bio', 'gene', 'cancer', 'breast', 'diabetes', 'disease',
      'medical', 'health', 'patient', 'cell', '医疗', '生物', '基因'], 'biology'),
    (['finance', 'fin', 'credit', 'loan', 'bank', 'income', 'census',
      'economic', '金融', '经济', '收入'], 'finance'),
    (['synthetic', 'syn', 'generate', 'make_class', 'make_reg',
      'artificial', '合成', '生成'], 'synthetic'),
    (['classification', 'class', 'classify', 'binary', 'multiclass',
      '分类', '二分类', '多分类'], 'classification'),
]


def infer_category(name, file_ext='csv'):
    """根据数据集名称推断分类"""
    name_lower = name.lower().replace('_', ' ').replace('-', ' ')
    for keywords, category in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw in name_lower:
                return category
    if file_ext in ('csv', 'xlsx', 'xls', 'parquet'):
        return 'tabular'
    return 'other'


def migrate():
    app = create_app()
    with app.app_context():
        print("=" * 60)
        print("数据集分类迁移: ENUM -> VARCHAR(50)")
        print("=" * 60)

        # 1. 修改表结构
        try:
            db.session.execute(db.text(
                "ALTER TABLE datasets MODIFY COLUMN category VARCHAR(50) "
                "DEFAULT 'other' NOT NULL"
            ))
            db.session.commit()
            print("[OK] 表结构已修改: category VARCHAR(50)")
        except Exception as e:
            db.session.rollback()
            print(f"[WARN] 表结构修改失败 (可能已是VARCHAR): {e}")

        # 2. 更新现有数据集的分类
        datasets = Dataset.query.all()
        print(f"\n现有数据集: {len(datasets)} 个\n")
        updated = 0

        for ds in datasets:
            old_cat = ds.category
            new_cat = infer_category(ds.name, ds.file_format)
            if old_cat != new_cat:
                ds.category = new_cat
                updated += 1
                print(f"  {ds.name:40s} {old_cat:10s} -> {new_cat}")
            else:
                print(f"  {ds.name:40s} {old_cat:10s}   (不变)")

        db.session.commit()
        print(f"\n[OK] 更新完成: {updated} 个数据集分类已更新")

        # 3. 统计新分类分布
        from collections import Counter
        cats = Counter(d.category for d in Dataset.query.all())
        print("\n分类分布:")
        for cat, count in cats.most_common():
            print(f"  {cat:20s} {count} 个")


if __name__ == '__main__':
    migrate()
