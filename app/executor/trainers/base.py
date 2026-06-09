"""
训练器抽象基类
定义所有训练器必须实现的接口
"""
import threading
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone

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

    # ============ 模板方法 (子类无需重写) ============

    def run(self):
        """训练主流程 — 模板方法

        子类可通过设置 self._early_stop = True 来触发早停 (在 train_epoch 中设置)
        子类可通过 self._best_val_metric 记录最佳验证指标供日志输出
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

            # 加载数据
            self.callback.on_log('正在加载数据...')
            self.load_data()
            self.callback.on_log('数据加载完成')

            # 构建模型
            self.callback.on_log('正在构建模型...')
            self.build_model()
            self.callback.on_log('模型构建完成')

            # 训练循环
            for epoch in range(self.total_epochs):
                # 检查取消
                if self._cancel_event.is_set():
                    self.callback.on_cancel()
                    return

                # 检查暂停
                self._pause_event.wait()

                # 检查早停
                if self._early_stop:
                    stopped_at = epoch
                    self.callback.on_log(
                        f'[早停] 验证指标连续未改善，在第 {stopped_at}/{self.total_epochs} 轮提前停止'
                    )
                    break

                # 训练一个 epoch
                metrics = self.train_epoch(epoch)
                self.callback.on_epoch_end(epoch, self.total_epochs, metrics)

                # 检查子类是否在 train_epoch 中设置了早停
                if self._early_stop:
                    stopped_at = epoch + 1
                    self.callback.on_log(
                        f'[早停] 验证指标连续未改善，在第 {stopped_at}/{self.total_epochs} 轮提前停止'
                    )
                    break

                # 日志 — 突出 val metrics
                train_items = {k: v for k, v in metrics.items()
                              if isinstance(v, float) and k.startswith('train_')}
                val_items = {k: v for k, v in metrics.items()
                            if isinstance(v, float) and k.startswith('val_')}
                other_items = {k: v for k, v in metrics.items()
                              if isinstance(v, float) and not k.startswith(('train_', 'val_'))}

                parts = []
                if train_items:
                    parts.append('train: ' + ' '.join(
                        f'{k[6:]}={v:.4f}' for k, v in train_items.items()))
                if val_items:
                    parts.append('val: ' + ' '.join(
                        f'{k[4:]}={v:.4f}' for k, v in val_items.items()))
                if other_items:
                    parts.append(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                                         for k, v in other_items.items()))
                self.callback.on_log(f'Epoch {epoch + 1}/{self.total_epochs} - {", ".join(parts)}')

            # 评估
            self.callback.on_log('正在最终评估...')
            final_metrics = self.evaluate()
            self.callback.on_log(f'最终指标: {final_metrics}')

            # 保存模型
            os.makedirs(self.output_dir, exist_ok=True)
            model_path = os.path.join(self.output_dir, 'model')
            self.save_model(model_path)

            self.callback.on_complete(final_metrics)

        except Exception as e:
            self.callback.on_error(str(e))
            raise

    # ============ 控制接口 ============

    def pause(self):
        """暂停训练"""
        self._pause_event.clear()

    def resume(self):
        """恢复训练"""
        self._pause_event.set()

    def cancel(self):
        """取消训练"""
        self._cancel_event.set()
        self._pause_event.set()  # 取消时也解除暂停状态

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()
