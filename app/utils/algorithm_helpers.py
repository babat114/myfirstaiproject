"""
集中式算法参数修复工具
消除 engine.py / training_service.py / hyperparameter_tuning.py 中分散的
KMeans algorithm 参数冲突修复逻辑。
"""
import json

from app import logger

# KMeans 构造参数中与 ML 算法名冲突的参数值
_KMEANS_ALGO_PARAMS = {'lloyd', 'elkan', 'auto', 'full'}


def fix_kmeans_algorithm(algorithm: str, hyperparams: dict = None, model=None) -> str:
    """
    检测并修正 KMeans 参数 algorithm=lloyd/elkan 与 ML 算法名 kmeans 的冲突。

    当超参的 algorithm 字段值为 KMeans 构造参数 (lloyd/elkan/auto/full) 而非
    ML 算法名 (kmeans) 时，自动修正为 "kmeans" 并持久化到关联 model 的
    hyperparameters_json。

    Args:
        algorithm: 当前 algorithm 值
        hyperparams: 超参数字典 (会被原地修改)
        model: 关联的 ModelRecord 实例 (会被原地修改 hyperparameters_json)

    Returns:
        修正后的 algorithm 值
    """
    if algorithm.lower() not in _KMEANS_ALGO_PARAMS:
        return algorithm

    logger.warning(
        f'检测到错误的algorithm值 "{algorithm}" (KMeans参数值), '
        f'自动修正为 "kmeans"'
    )
    corrected = 'kmeans'

    # 原地修复 hyperparams 字典
    if hyperparams is not None:
        hyperparams['algorithm'] = corrected

    # 持久化修复到 model
    if model is not None:
        try:
            hp = {}
            if model.hyperparameters_json:
                hp = json.loads(model.hyperparameters_json)
            hp['algorithm'] = corrected
            model.hyperparameters_json = json.dumps(hp, ensure_ascii=False)
        except Exception:
            pass

    return corrected
