"""
============================================
数据库初始化脚本
创建所有表结构
运行: python scripts/init_db.py
============================================
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import User, Dataset, ModelRecord, TrainingJob


def init_database():
    """创建所有数据库表"""
    app = create_app()

    with app.app_context():
        print("正在创建数据库表...")

        # 创建所有表
        db.create_all()

        print("✅ 数据库表创建完成!")
        print("\n创建的表:")
        for table in db.metadata.sorted_tables:
            print(f"  📋 {table.name}")


if __name__ == '__main__':
    init_database()
