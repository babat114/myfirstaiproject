"""
============================================
回填模型描述脚本 (v2)
为所有现有模型生成增强描述:
  应用场景：基于"{数据集}"的{任务}模型...
  使用方式：输入{特征说明}，模型输出{输出说明}。
  算法原理描述

用法: python scripts/backfill_model_descriptions.py
============================================
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.utils.algorithm_info import ALGORITHM_INFO
from app.services.model_recommender import generate_enhanced_description


def main():
    app = create_app()
    with app.app_context():
        models = ModelRecord.query.all()
        updated = 0
        unchanged = 0

        for m in models:
            hp = m.hyperparameters_dict
            algorithm = hp.get('algorithm', '')
            task_type = hp.get('task_type', m.model_type or '')
            target_column = hp.get('target_column', '')

            # 数据集名称
            dataset_name = ''
            if m.training_dataset and m.training_dataset.name:
                dataset_name = m.training_dataset.name
            else:
                name_parts = m.name.split(' - ')[0].split(' (')[0].strip()
                if name_parts and len(name_parts) >= 2:
                    dataset_name = name_parts
                else:
                    dataset_name = m.name

            # 尝试从数据集 summary 获取特征名和类别
            feature_names = []
            class_labels = []
            if m.training_dataset and m.training_dataset.summary_json:
                try:
                    import json as _json
                    summary = _json.loads(m.training_dataset.summary_json) if isinstance(m.training_dataset.summary_json, str) else m.training_dataset.summary_json
                    cols = summary.get('columns', [])
                    if target_column and target_column in cols:
                        feature_names = [c for c in cols if c != target_column]
                    elif cols:
                        feature_names = cols
                except Exception:
                    pass

            new_desc = generate_enhanced_description(
                dataset_name=dataset_name,
                task_type=task_type,
                algorithm=algorithm,
                class_labels=class_labels,
                feature_names=feature_names,
                target_column=target_column,
                model_name=m.name,
            )

            if new_desc != m.description:
                print(f'  [{m.id}] {m.name}')
                old_preview = (m.description or '')[:80].replace('\n', ' ')
                print(f'    ├ 旧: {old_preview}...')
                new_first = new_desc.split('\n')[0]
                print(f'    └ 新: {new_first}')
                m.description = new_desc
                updated += 1
            else:
                unchanged += 1

        db.session.commit()
        print(f'\n完成: 更新 {updated} 个模型描述, {unchanged} 个未变化')


if __name__ == '__main__':
    main()
