"""
训练器包
每种训练器实现一个具体的机器学习框架
"""
from app.executor.trainers.base import BaseTrainer
from app.executor.trainers.sklearn_trainer import SklearnTrainer

__all__ = ['BaseTrainer', 'SklearnTrainer']
