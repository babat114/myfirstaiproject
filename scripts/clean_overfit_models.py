"""
清理准确率100%的过拟合模型。

删除标准：accuracy >= 0.999 (即99.9%以上，基本上都是过拟合)
同时删除：
1. 数据库中的 ModelRecord
2. 关联的 TrainingJob
3. 实验目录 (experiments/<uuid>/)
"""
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.training_job import TrainingJob

THRESHOLD = 0.999  # 准确率 >= 99.9% 视为过拟合

def main():
    app = create_app()
    with app.app_context():
        # 1. 查找所有过拟合模型
        models = ModelRecord.query.all()
        overfit = [m for m in models if m.accuracy is not None and m.accuracy >= THRESHOLD]

        print(f"=" * 70)
        print(f"  过拟合模型清理工具")
        print(f"  阈值: accuracy >= {THRESHOLD} ({THRESHOLD*100:.1f}%)")
        print(f"  待删除: {len(overfit)} 个模型 (共 {len(models)} 个)")
        print(f"=" * 70)

        if not overfit:
            print("\n没有找到过拟合模型，退出。")
            return

        # 2. 分类显示
        by_dataset = {}
        for m in overfit:
            ds_name = "unknown"
            if m.training_dataset_id:
                from app.models.dataset import Dataset
                ds = Dataset.query.get(m.training_dataset_id)
                ds_name = ds.name if ds else f"dataset_id={m.training_dataset_id}"
            by_dataset.setdefault(ds_name, []).append(m)

        print("\n按数据集分组:")
        for ds_name, model_list in sorted(by_dataset.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{len(model_list)}个] {ds_name}:")
            for m in model_list:
                print(f"    ID={m.id:>4} | acc={m.accuracy:.4f} | f1={m.f1_score:.4f} | "
                      f"name={m.name[:50]} | uuid={m.uuid}")

        # 3. 汇总
        total_experiments_to_delete = len(set(m.uuid for m in overfit if m.uuid))
        print(f"\n" + "-" * 70)
        print(f"总计: {len(overfit)} 个模型记录将被删除")
        print(f"涉及: {total_experiments_to_delete} 个实验目录")
        print(f"-" * 70)

        # 4. 确认
        force = '--force' in sys.argv or '-f' in sys.argv
        if force:
            print("\n[!] --force 模式：跳过确认，直接删除！")
        else:
            print("\n[!] 此操作将从数据库和磁盘永久删除以上模型！")
            confirm = input("输入 'yes' 确认删除，其他任意键取消: ").strip()
            if confirm.lower() != 'yes':
                print("已取消。")
                return

        # 5. 删除（使用原生SQL绕过ORM循环外键依赖）
        from sqlalchemy import text
        experiments_dir = Path(__file__).resolve().parent.parent / 'experiments'
        deleted_models = 0
        deleted_jobs = 0
        deleted_dirs = 0
        errors = []

        # 收集模型ID和关联的job_id、uuid
        model_ids = []
        job_ids = set()
        uuid_set = set()
        for m in overfit:
            model_ids.append(m.id)
            if m.training_job_id:
                job_ids.add(m.training_job_id)
            if m.uuid:
                uuid_set.add(m.uuid)

        # 5a. 先删除实验目录（磁盘操作，与DB无关）
        for uuid in uuid_set:
            exp_dir = experiments_dir / uuid
            if exp_dir.exists():
                try:
                    shutil.rmtree(exp_dir)
                    deleted_dirs += 1
                    print(f"  [OK] 删除目录: experiments/{uuid}")
                except Exception as e:
                    errors.append(f"目录删除失败 {uuid}: {e}")

        # 5b. 断开循环外键 + 删除（关闭FK检查）
        try:
            # MySQL 需要先关闭外键检查
            db.session.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

            # 断开 model_records.training_job_id -> NULL
            result = db.session.execute(
                text("UPDATE model_records SET training_job_id = NULL WHERE id IN :ids"),
                {"ids": tuple(model_ids)}
            )
            print(f"  [OK] 断开 {result.rowcount} 个 model_records.training_job_id")

            # 断开 training_jobs.model_id -> NULL
            result = db.session.execute(
                text("UPDATE training_jobs SET model_id = NULL WHERE id IN :ids"),
                {"ids": tuple(job_ids)}
            )
            print(f"  [OK] 断开 {result.rowcount} 个 training_jobs.model_id")

            # 删除 training_jobs
            result = db.session.execute(
                text("DELETE FROM training_jobs WHERE id IN :ids"),
                {"ids": tuple(job_ids)}
            )
            deleted_jobs = result.rowcount
            print(f"  [OK] 删除 {deleted_jobs} 个 training_jobs")

            # 删除 model_records
            result = db.session.execute(
                text("DELETE FROM model_records WHERE id IN :ids"),
                {"ids": tuple(model_ids)}
            )
            deleted_models = result.rowcount
            print(f"  [OK] 删除 {deleted_models} 个 model_records")

            # 恢复外键检查
            db.session.execute(text("SET FOREIGN_KEY_CHECKS = 1"))

        except Exception as e:
            db.session.execute(text("SET FOREIGN_KEY_CHECKS = 1"))  # 恢复
            errors.append(f"DB操作失败: {e}")

        # 提交事务
        try:
            db.session.commit()
            print(f"\n[OK] 数据库提交成功")
        except Exception as e:
            db.session.rollback()
            print(f"\n[FAIL] 数据库提交失败: {e}")
            errors.append(f"数据库提交: {e}")

        # 6. 报告
        print(f"\n{'=' * 70}")
        print(f"  清理完成")
        print(f"  删除模型记录: {deleted_models}")
        print(f"  删除训练任务: {deleted_jobs}")
        print(f"  删除实验目录: {deleted_dirs}")
        if errors:
            print(f"  错误: {len(errors)}")
            for e in errors:
                print(f"    - {e}")
        print(f"{'=' * 70}")

        # 7. 验证剩余模型
        remaining = ModelRecord.query.count()
        still_overfit = ModelRecord.query.filter(
            ModelRecord.accuracy >= THRESHOLD
        ).count()
        print(f"\n剩余模型总数: {remaining}")
        print(f"剩余过拟合模型: {still_overfit}")

if __name__ == '__main__':
    main()
