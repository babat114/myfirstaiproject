"""
训练执行引擎包
提供基于线程池的轻量级训练执行、暂停、取消功能
"""

from app.executor.engine import TrainingExecutor, get_executor
from app.executor.trainers.base import BaseTrainer
from app.executor.trainers.sklearn_trainer import SklearnTrainer

__all__ = ['TrainingExecutor', 'get_executor', 'BaseTrainer', 'SklearnTrainer']
