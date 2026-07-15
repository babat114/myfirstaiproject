"""
训练器抽象基类
定义所有训练器必须实现的接口
"""

import os
import threading
from abc import ABC, abstractmethod

from app.executor.callbacks import TrainingCallback


class BaseTrainer(ABC):
    """训练器抽象基类 — 使用模板方法 (Template Method) 设计模式

    子类必须实现5个抽象方法:
        1. load_data()        — 加载并预处理数据集
        2. build_model()      — 构建模型 (sklearn 或 PyTorch)
        3. train_epoch(epoch) — 训练一个epoch, 返回指标字典
        4. evaluate()         — 在测试集上最终评估
        5. save_model(path)   — 保存模型到指定路径

    模板方法 run() 的执行流程:
        load_data → build_model → [train_epoch × N] → evaluate → save_model

    线程控制:
        _pause_event  — 暂停信号 (clear=暂停, set=继续)
        _cancel_event — 取消信号 (set=停止训练)

    回调系统:
        self.callback (TrainingCallback) — 自动更新数据库进度/日志/指标
    """

    def __init__(self, job, dataset, hyperparams: dict = None):
        """
        Args:
            job: TrainingJob 模型实例
            dataset: Dataset 模型实例
            hyperparams: 超参数字典
        """
        self.job = job
        self.dataset = dataset
        self.hyperparams = hyperparams or {}
        self.callback = TrainingCallback(job.id)

        # 中断控制
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始状态: 不暂停
        self._cancel_event = threading.Event()

        # 训练参数
        self.total_epochs = job.total_epochs or self.hyperparams.get('epochs', 10)
        self.output_dir = os.path.join('experiments', job.uuid)
        # 检查点: 每N个epoch保存一次训练快照 (0 = 禁用)
        self.checkpoint_frequency = int(self.hyperparams.get('checkpoint_frequency', 5))
        self._current_epoch = 0  # 当前epoch (用于断点续训)
        self._last_train_metrics = {}  # 最后一轮训练指标, 用于合并到final_metrics

    # ============ 子类必须实现 ============

    @abstractmethod
    def load_data(self):
        """加载并预处理数据集，返回 (X_train, X_test, y_train, y_test)"""
        ...

    @abstractmethod
    def build_model(self):
        """构建模型"""
        ...

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        """训练一个 epoch，返回指标字典如 {'loss': 0.5, 'accuracy': 0.8}"""
        ...

    @abstractmethod
    def evaluate(self) -> dict:
        """在测试集上评估，返回指标字典"""
        ...

    @abstractmethod
    def save_model(self, path: str):
        """保存模型到指定路径"""
        ...

    # ============ 交叉验证 (子类可选重写) ============

    def run_cross_validation(self) -> dict | None:
        """交叉验证评估泛化能力 (默认 no-op, sklearn 子类重写)

        Returns:
            dict with cv_mean, cv_std, cv_folds, n_samples, error | None
        """
        return None

    # ============ 检查点 (子类可选重写) ============

    def save_checkpoint(self):  # noqa: B027
        pass

    @staticmethod
    def load_checkpoint(output_dir: str) -> dict:
        """从磁盘加载训练快照。返回恢复状态字典, 无检查点时返回空字典。

        子类可重写以恢复框架特定状态。返回的字典可包含:
            - 'epoch': 已完成的epoch数 (从此epoch继续)
            - 'best_val_loss': 最佳验证损失
            - 'patience_counter': 早停计数器
            - '_restore': 任意子类特定数据 (传给 restore_checkpoint)
        """
        return {}

    def restore_checkpoint(self, ckpt: dict):  # noqa: B027
        pass

    @staticmethod
    def has_checkpoint(output_dir: str) -> bool:
        """检查是否存在训练快照"""
        return False

    # ============ 模板方法 (子类无需重写) ============

    def run(self):
        """训练主流程 — 模板方法

        子类可通过设置 self._early_stop = True 来触发早停 (在 train_epoch 中设置)
        子类可通过 self._best_val_metric 记录最佳验证指标供日志输出

        v2: 支持检查点持久化 — 每N轮保存训练快照, 崩溃后可恢复
        """
        # 早停标志 (子类在 train_epoch 中设置)
        if not hasattr(self, '_early_stop'):
            self._early_stop = False
        if not hasattr(self, '_best_val_metric'):
            self._best_val_metric = None

        try:
            self.callback.on_start()
            self.callback.on_log(f'框架: {self.__class__.__name__}')
            self.callback.on_log(f'超参数: {self.hyperparams}')

            # —— 检查是否有检查点可恢复 ——
            os.makedirs(self.output_dir, exist_ok=True)
            ckpt_meta = None
            if self.has_checkpoint(self.output_dir):
                ckpt_meta = self.load_checkpoint(self.output_dir)
                self._current_epoch = ckpt_meta.get('epoch', 0)
                self.callback.on_log(f'[检查点] 从 epoch {self._current_epoch}/{self.total_epochs} 恢复训练')
            else:
                self._current_epoch = 0

            # 加载数据
            self.callback.on_log('正在加载数据...')
            self.load_data()
            self.callback.on_log('数据加载完成')

            # 构建模型
            self.callback.on_log('正在构建模型...')
            self.build_model()
            self.callback.on_log('模型构建完成')

            # —— 恢复检查点权重/优化器/调度器状态 ——
            if ckpt_meta:
                self.restore_checkpoint(ckpt_meta)
                self.callback.on_log('[检查点] 权重/优化器/早停状态已恢复')

            # 训练循环
            for epoch in range(self._current_epoch, self.total_epochs):
                self._current_epoch = epoch

                # 检查取消
                if self._cancel_event.is_set():
                    self.callback.on_cancel()
                    return

                # 检查暂停
                self._pause_event.wait()

                # 检查早停
                if self._early_stop:
                    stopped_at = epoch
                    self.callback.on_log(f'[早停] 验证指标连续未改善，在第 {stopped_at}/{self.total_epochs} 轮提前停止')
                    break

                # 训练一个 epoch
                metrics = self.train_epoch(epoch)
                self._last_train_metrics = metrics  # 保存最后一轮训练指标
                self.callback.on_epoch_end(epoch, self.total_epochs, metrics)

                # 检查子类是否在 train_epoch 中设置了早停
                if self._early_stop:
                    stopped_at = epoch + 1
                    self.callback.on_log(f'[早停] 验证指标连续未改善，在第 {stopped_at}/{self.total_epochs} 轮提前停止')
                    break

                # —— 定期保存检查点 ——
                if self.checkpoint_frequency > 0 and (epoch + 1) % self.checkpoint_frequency == 0:
                    self.save_checkpoint()
                    self.callback.on_log(f'[检查点] epoch {epoch + 1}/{self.total_epochs} 快照已保存')

                # 日志 — 突出 val metrics
                train_items = {k: v for k, v in metrics.items() if isinstance(v, float) and k.startswith('train_')}
                val_items = {k: v for k, v in metrics.items() if isinstance(v, float) and k.startswith('val_')}
                other_items = {
                    k: v for k, v in metrics.items() if isinstance(v, float) and not k.startswith(('train_', 'val_'))
                }

                parts = []
                if train_items:
                    parts.append('train: ' + ' '.join(f'{k[6:]}={v:.4f}' for k, v in train_items.items()))
                if val_items:
                    parts.append('val: ' + ' '.join(f'{k[4:]}={v:.4f}' for k, v in val_items.items()))
                if other_items:
                    parts.append(
                        ' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in other_items.items())
                    )
                self.callback.on_log(f'Epoch {epoch + 1}/{self.total_epochs} - {", ".join(parts)}')

            # 评估
            self.callback.on_log('正在最终评估...')
            final_metrics = self.evaluate()

            # 合并最后一轮训练指标 (train_accuracy/train_loss), 保留到final_metrics
            # 使diagnose_model_quality.py的train_test_gap检测能正常工作
            for key in ('train_accuracy', 'train_loss', 'train_precision', 'train_recall', 'train_f1'):
                if key in self._last_train_metrics:
                    final_metrics[key] = self._last_train_metrics[key]

            self.callback.on_log(f'最终指标: {final_metrics}')

            # 交叉验证 (sklearn子类实现, 其他框架no-op)
            cv_results = self.run_cross_validation()
            if cv_results and not cv_results.get('error'):
                self.callback.on_log(
                    f'CV: mean={cv_results["cv_mean"]:.4f} +/- {cv_results["cv_std"]:.4f} '
                    f'({cv_results["cv_folds"]} folds, n={cv_results["n_samples"]})'
                )
                final_metrics['cv_mean'] = cv_results.get('cv_mean')
                final_metrics['cv_std'] = cv_results.get('cv_std')
                final_metrics['cv_folds'] = cv_results.get('cv_folds')
            elif cv_results and cv_results.get('error'):
                self.callback.on_log(f'[CV] 跳过: {cv_results["error"]}')

            # 保存模型
            os.makedirs(self.output_dir, exist_ok=True)
            model_path = os.path.join(self.output_dir, 'model')
            self.save_model(model_path)

            # —— 训练完成, 清理检查点 ——
            self._cleanup_checkpoint()

            self.callback.on_complete(final_metrics)

        except Exception as e:
            # —— 异常时保存检查点以便恢复 ——
            self.callback.on_log('[检查点] 训练异常, 正在保存快照...')
            try:
                self.save_checkpoint()
            except Exception as ckpt_err:
                self.callback.on_log(f'[检查点] 快照保存失败: {ckpt_err}')
            self.callback.on_error(str(e))
            raise

    # ============ 控制接口 ============

    def pause(self):
        """暂停训练 — 自动保存检查点"""
        # 先保存检查点再暂停
        try:
            self.save_checkpoint()
            self.callback.on_log(f'[检查点] 训练暂停, epoch {self._current_epoch + 1} 快照已保存')
        except Exception as e:
            self.callback.on_log(f'[检查点] 暂停快照保存失败: {e}')
        self._pause_event.clear()

    def resume(self):
        """恢复训练"""
        self._pause_event.set()

    def cancel(self):
        """取消训练"""
        self._cancel_event.set()
        self._pause_event.set()  # 取消时也解除暂停状态

    def _cleanup_checkpoint(self):
        """训练成功完成或取消后清理检查点文件 (文件或目录)"""
        import glob
        import shutil

        pattern = os.path.join(self.output_dir, 'checkpoint*')
        for f in glob.glob(pattern):
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
            except OSError:
                pass

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()
