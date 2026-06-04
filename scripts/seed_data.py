"""
============================================
种子数据脚本
填充示例数据用于开发和测试
运行: python scripts/seed_data.py
============================================
"""
import sys
import os
import random
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, Dataset, ModelRecord, TrainingJob
from app.services.auth_service import AuthService


def seed_database():
    """填充示例数据"""
    app = create_app()

    with app.app_context():
        print("正在填充示例数据...")

        # 清空现有数据 (开发环境)
        print("清空现有数据...")
        TrainingJob.query.delete()
        ModelRecord.query.delete()
        Dataset.query.delete()
        User.query.delete()
        db.session.commit()

        # 创建管理员
        admin, err = AuthService.register(
            username='admin',
            email='admin@aiplatform.com',
            password='Admin123456',
            full_name='系统管理员',
        )
        if admin:
            admin.role = 'admin'
            admin.is_verified = True
            admin.bio = '平台管理员账户'
            admin.organization = 'AI Platform Team'
            print(f"✅ 管理员: admin / Admin123456")

        # 创建普通用户
        users = []
        sample_users = [
            ('researcher1', 'researcher1@ai.com', '张三', 'AI Lab'),
            ('researcher2', 'researcher2@ai.com', '李四', '数据科学部'),
            ('viewer1', 'viewer1@ai.com', '王五', '产品部'),
        ]
        for uname, email, full_name, org in sample_users:
            user, err = AuthService.register(
                username=uname, email=email,
                password='User123456', full_name=full_name,
            )
            if user:
                user.organization = org
                user.role = 'viewer' if 'viewer' in uname else 'researcher'
                user.is_verified = True
                users.append(user)
                print(f"  ✅ {uname} ({full_name})")

        db.session.commit()

        # 创建示例数据集
        datasets_data = [
            {'name': 'MNIST 手写数字', 'description': '经典手写数字识别数据集，包含60000张训练图片和10000张测试图片',
             'category': 'image', 'format': 'csv', 'size': 11500000, 'rows': 70000, 'cols': 785, 'public': True},
            {'name': 'IMDB 电影评论', 'description': 'IMDB 电影评论情感分析数据集',
             'category': 'text', 'format': 'csv', 'size': 8500000, 'rows': 50000, 'cols': 2, 'public': True},
            {'name': '房价预测', 'description': '波士顿房价回归分析数据集',
             'category': 'tabular', 'format': 'csv', 'size': 2500000, 'rows': 10000, 'cols': 15, 'public': False},
            {'name': 'CIFAR-10 图像分类', 'description': '10类别小型彩色图像数据集',
             'category': 'image', 'format': 'parquet', 'size': 50000000, 'rows': 60000, 'cols': 3072, 'public': True},
        ]

        datasets = []
        for i, ds_data in enumerate(datasets_data):
            owner = users[i % len(users)]
            ds = Dataset(
                name=ds_data['name'],
                description=ds_data['description'],
                file_path=f'/data/{ds_data["name"].replace(" ", "_")}.{ds_data["format"]}',
                file_size=ds_data['size'],
                file_format=ds_data['format'],
                category=ds_data['category'],
                row_count=ds_data['rows'],
                column_count=ds_data['cols'],
                is_public=ds_data['public'],
                owner_id=owner.id,
                status='ready',
            )
            db.session.add(ds)
            datasets.append(ds)
            print(f"  📊 {ds_data['name']}")

        db.session.commit()

        # 创建示例模型
        models_data = [
            {'name': 'CNN-MNIST', 'type': 'classification', 'framework': 'PyTorch',
             'accuracy': 0.9932, 'precision': 0.992, 'recall': 0.991, 'f1': 0.9915, 'status': 'deployed'},
            {'name': 'BERT 情感分析', 'type': 'nlp', 'framework': 'TensorFlow',
             'accuracy': 0.9245, 'precision': 0.918, 'recall': 0.925, 'f1': 0.9215, 'status': 'deployed'},
            {'name': 'ResNet-50 图像分类', 'type': 'computer_vision', 'framework': 'PyTorch',
             'accuracy': 0.9523, 'precision': 0.950, 'recall': 0.948, 'f1': 0.9490, 'status': 'trained'},
            {'name': 'XGBoost 房价预测', 'type': 'regression', 'framework': 'scikit-learn',
             'accuracy': 0.8920, 'precision': None, 'recall': None, 'f1': None, 'status': 'deployed'},
            {'name': 'GPT-2 文本生成', 'type': 'generative', 'framework': 'PyTorch',
             'accuracy': None, 'precision': None, 'recall': None, 'f1': None, 'status': 'draft'},
            {'name': 'LSTM 序列预测', 'type': 'regression', 'framework': 'Keras',
             'accuracy': 0.8756, 'precision': None, 'recall': None, 'f1': None, 'status': 'archived'},
        ]

        model_records = []
        for i, m_data in enumerate(models_data):
            owner = users[i % len(users)]
            ds = datasets[i % len(datasets)] if datasets else None

            m = ModelRecord(
                name=m_data['name'],
                description=f"{m_data['name']} 模型 - 使用{m_data['framework']}训练",
                version=f"1.{i}.0",
                model_type=m_data['type'],
                framework=m_data['framework'],
                accuracy=m_data['accuracy'],
                precision=m_data['precision'],
                recall=m_data['recall'],
                f1_score=m_data['f1'],
                status=m_data['status'],
                is_public=True,
                owner_id=owner.id,
                training_dataset_id=ds.id if ds else None,
            )
            db.session.add(m)
            model_records.append(m)
            print(f"  🎓 {m_data['name']}")

        db.session.commit()

        # 创建训练任务
        jobs_data = [
            {'name': 'CNN-MNIST 训练', 'type': 'training', 'status': 'completed', 'progress': 100.0},
            {'name': 'BERT 微调任务', 'type': 'fine_tuning', 'status': 'running', 'progress': 65.5},
            {'name': 'ResNet 评估', 'type': 'evaluation', 'status': 'queued', 'progress': 0.0},
        ]

        for i, j_data in enumerate(jobs_data):
            owner = users[i % len(users)]
            ds = datasets[i % len(datasets)] if datasets else None

            j = TrainingJob(
                name=j_data['name'],
                description=f'{j_data["name"]} - {j_data["type"]}',
                task_type=j_data['type'],
                framework='PyTorch',
                status=j_data['status'],
                progress_percent=j_data['progress'],
                total_epochs=50,
                total_steps=5000,
                current_epoch=33 if j_data['status'] == 'running' else 0,
                gpu_count=1,
                cpu_cores=4,
                memory_gb=8.0,
                owner_id=owner.id,
                dataset_id=ds.id if ds else None,
            )

            if j_data['status'] == 'completed':
                j.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
                j.completed_at = datetime.now(timezone.utc)
                j.log_text = '[2026-06-02 10:00:00] 训练开始\n[2026-06-02 11:30:00] Epoch 25/50 - loss: 0.123\n[2026-06-02 12:00:00] 训练完成 - accuracy: 0.993'
            elif j_data['status'] == 'running':
                j.started_at = datetime.now(timezone.utc) - timedelta(hours=1)
                j.log_text = '[2026-06-02 11:00:00] 训练开始\n[2026-06-02 11:30:00] Epoch 15/50 - loss: 0.245'

            db.session.add(j)
            print(f"  ▶️  {j_data['name']}")

        db.session.commit()

        print("\n✅ 示例数据填充完成!")
        print(f"""
数据统计:
  用户: {db.session.query(User).count()}
  数据集: {db.session.query(Dataset).count()}
  模型: {db.session.query(ModelRecord).count()}
  训练任务: {db.session.query(TrainingJob).count()}

登录信息:
  管理员: admin / Admin123456
  用户: researcher1 / User123456
""")

        db.session.commit()


if __name__ == '__main__':
    seed_database()
