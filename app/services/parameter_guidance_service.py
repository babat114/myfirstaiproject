"""
============================================
训练参数调整引导服务
分析训练结果指标，给出参数调优建议
============================================
"""
import json
import numpy as np
from typing import Optional
from app import logger


class ParameterGuidanceService:
    """训练参数引导引擎 — 分析训练结果 + 数据集特征，给出调参建议

    分析维度:
        1. 过拟合检测: train_acc >> test_acc → 增加正则化/降dropout/减epochs
        2. 欠拟合检测: train_acc和test_acc都低 → 增加复杂度/增epochs/换算法
        3. 学习率: loss震荡 → 降lr; loss下降慢 → 升lr
        4. 批大小: 内存够但收敛慢 → 增batch; loss跳跃 → 减batch
        5. 网络宽度: 长期不收敛 → 加宽hidden_layers
        6. 类不平衡: 某些类F1很低 → class_weight/重采样
    """

    @staticmethod
    def analyze_results(metrics_history: list, final_metrics: dict,
                        dataset_info: dict = None,
                        hyperparams: dict = None) -> dict:
        """全面分析训练结果并给出调参建议

        Args:
            metrics_history: epoch级指标历史 [{epoch, loss, accuracy, ...}, ...]
            final_metrics: 最终指标 {accuracy, loss, precision, recall, f1_score, ...}
            dataset_info: 数据集分析结果 (可选, 用于上下文)
            hyperparams: 使用的超参数 (可选, 用于针对性建议)

        Returns:
            {
                'diagnosis': str,       # 诊断结论
                'issues': [...],        # 发现的问题列表
                'suggestions': [...],   # 参数调整建议 (按优先级排序)
                'health_score': float,  # 训练健康度评分 0-100
                'next_steps': [...]     # 建议的下一步操作
            }
        """
        metrics_history = metrics_history or []
        final_metrics = final_metrics or {}
        hyperparams = hyperparams or {}

        issues = []
        suggestions = []
        health_deduct = 0

        # 确定任务类型
        task_type = 'classification'
        if 'r2' in final_metrics or 'mse' in final_metrics:
            task_type = 'regression'

        # === 1. 过拟合检测 ===
        overfit_info = ParameterGuidanceService._detect_overfitting(
            metrics_history, task_type
        )
        if overfit_info['overfitting']:
            severity = overfit_info['severity']
            health_deduct += 15 if severity == 'high' else 8
            issues.append({
                'type': 'overfitting',
                'severity': severity,
                'message': f"检测到{'严重' if severity == 'high' else '轻微'}过拟合"
                          f"(训练集表现远好于测试集)",
            })
            suggestions.append({
                'priority': 1,
                'param': 'dropout',
                'action': 'increase',
                'current': hyperparams.get('dropout', 0.3),
                'suggested': min(0.6, hyperparams.get('dropout', 0.3) + 0.2),
                'reason': '增加Dropout可有效防止过拟合',
                'alternative': '也可减少epochs或增加weight_decay',
            })
            suggestions.append({
                'priority': 2,
                'param': 'epochs',
                'action': 'decrease',
                'suggested_reason': '减少训练轮数，避免过度记忆训练数据',
            })

        # === 2. 欠拟合检测 ===
        if ParameterGuidanceService._detect_underfitting(metrics_history, final_metrics, task_type):
            health_deduct += 12
            issues.append({
                'type': 'underfitting',
                'severity': 'medium',
                'message': '模型可能欠拟合 — 训练集和测试集指标都偏低',
            })
            suggestions.append({
                'priority': 1,
                'param': 'hidden_layers',
                'action': 'increase',
                'current': hyperparams.get('hidden_layers', [128, 64]),
                'suggested': [256, 128, 64, 32],
                'reason': '增加网络深度/宽度，提升模型容量',
            })
            suggestions.append({
                'priority': 2,
                'param': 'epochs',
                'action': 'increase',
                'suggested_reason': '增加训练轮数，给模型更多学习机会',
            })
            suggestions.append({
                'priority': 3,
                'param': 'algorithm',
                'action': 'change',
                'suggested': 'gradient_boosting',
                'reason': '可尝试更强大的算法如Gradient Boosting',
            })

        # === 3. Loss 曲线分析 ===
        loss_analysis = ParameterGuidanceService._analyze_loss_curve(metrics_history)
        if loss_analysis['oscillating']:
            health_deduct += 10
            issues.append({
                'type': 'loss_oscillation',
                'severity': 'medium',
                'message': 'Loss曲线震荡剧烈 — 学习率可能过大',
            })
            suggestions.append({
                'priority': 1,
                'param': 'learning_rate',
                'action': 'decrease',
                'current': hyperparams.get('learning_rate', 0.001),
                'suggested': hyperparams.get('learning_rate', 0.001) / 5,
                'reason': '降低学习率可稳定收敛',
            })
        elif loss_analysis['slow_convergence']:
            health_deduct += 5
            suggestions.append({
                'priority': 3,
                'param': 'learning_rate',
                'action': 'increase',
                'suggested': hyperparams.get('learning_rate', 0.001) * 3,
                'reason': '可适当提高学习率加速收敛',
            })

        # === 4. 分类特定分析 ===
        if task_type == 'classification':
            class_issues = ParameterGuidanceService._analyze_classification(final_metrics)
            issues.extend(class_issues['issues'])
            health_deduct += class_issues['deduction']
            suggestions.extend(class_issues['suggestions'])

        # === 5. 回归特定分析 ===
        if task_type == 'regression':
            reg_issues = ParameterGuidanceService._analyze_regression(final_metrics)
            issues.extend(reg_issues['issues'])
            health_deduct += reg_issues['deduction']
            suggestions.extend(reg_issues['suggestions'])

        # === 6. 生成综合诊断 ===
        health_score = max(0, 100 - health_deduct)

        if health_score >= 85:
            diagnosis = "🎉 训练状态良好！模型表现优秀，无需大幅调整。"
        elif health_score >= 65:
            diagnosis = "👍 训练状态尚可，有几处可以优化以进一步提升。"
        elif health_score >= 40:
            diagnosis = "⚠️ 存在一些问题需要关注，建议按下方建议调整。"
        else:
            diagnosis = "🔴 训练状态较差，建议重新审视数据处理和模型选择。"

        # 去重 + 按优先级排序
        seen = set()
        unique_suggestions = []
        for s in sorted(suggestions, key=lambda x: x.get('priority', 99)):
            key = s.get('param', '') + s.get('action', '')
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        # 生成下一步操作
        next_steps = ParameterGuidanceService._generate_next_steps(
            health_score, issues, unique_suggestions, hyperparams
        )

        return {
            'diagnosis': diagnosis,
            'health_score': health_score,
            'issues': issues,
            'suggestions': unique_suggestions[:8],
            'next_steps': next_steps,
            'task_type': task_type,
        }

    @staticmethod
    def _detect_overfitting(history, task_type) -> dict:
        """检测过拟合"""
        if len(history) < 3:
            return {'overfitting': False, 'severity': 'none'}

        # 取前1/3和后1/3的指标对比
        n = len(history)
        early = history[:max(1, n // 3)]
        late = history[-max(1, n // 3):]

        def get_score(m):
            if task_type == 'regression':
                return m.get('r2', m.get('train_r2', 0))
            return m.get('accuracy', m.get('train_accuracy', 0))

        early_scores = [get_score(m) for m in early if get_score(m) is not None]
        late_scores = [get_score(m) for m in late if get_score(m) is not None]

        if not early_scores or not late_scores:
            return {'overfitting': False, 'severity': 'none'}

        # 训练指标持续上升但loss不降 → 过拟合迹象
        score_rising = np.mean(late_scores) > np.mean(early_scores) * 1.15

        # 检查train/test gap (如果有)
        gap_widening = False
        for m in history:
            train_acc = m.get('train_accuracy', m.get('accuracy', 0))
            test_acc = m.get('test_accuracy', m.get('val_accuracy', 0))
            if train_acc and test_acc and (train_acc - test_acc) > 0.1:
                gap_widening = True
                break

        if score_rising and gap_widening:
            return {'overfitting': True, 'severity': 'high'}
        elif score_rising:
            return {'overfitting': True, 'severity': 'medium'}
        return {'overfitting': False, 'severity': 'none'}

    @staticmethod
    def _detect_underfitting(history, final_metrics, task_type) -> bool:
        """检测欠拟合"""
        if task_type == 'classification':
            acc = final_metrics.get('accuracy', 0)
            return acc < 0.60 and acc > 0
        else:
            r2 = final_metrics.get('r2', 0)
            return r2 < 0.30

    @staticmethod
    def _analyze_loss_curve(history) -> dict:
        """分析Loss曲线"""
        if len(history) < 5:
            return {'oscillating': False, 'slow_convergence': False}

        losses = [m.get('loss', m.get('train_loss', 0)) for m in history if 'loss' in m or 'train_loss' in m]
        if len(losses) < 5:
            return {'oscillating': False, 'slow_convergence': False}

        # 检测震荡: 相邻epoch loss变化超过前一个epoch的30%
        changes = [abs(losses[i] - losses[i - 1]) for i in range(1, len(losses))]
        avg_loss = np.mean(losses)
        oscillating = any(c > avg_loss * 0.5 for c in changes if avg_loss > 0)

        # 检测收敛速度: 最后几个epoch loss仍然较高
        slow = losses[-1] > losses[0] * 0.7 if losses[0] > 0 else False

        return {'oscillating': oscillating, 'slow_convergence': slow}

    @staticmethod
    def _analyze_classification(metrics) -> dict:
        """分类特定分析"""
        issues = []
        suggestions = []
        deduction = 0

        # 各类F1差异大 → 类别不平衡
        if 'f1_score' in metrics:
            f1 = metrics['f1_score']
            if f1 < 0.50:
                deduction += 10
                issues.append({
                    'type': 'low_f1', 'severity': 'medium',
                    'message': f'F1分数偏低 ({f1:.2%})，可能存在类别不平衡',
                })
                suggestions.append({
                    'priority': 1, 'param': 'class_weight',
                    'action': 'set', 'suggested': 'balanced',
                    'reason': '使用balanced权重处理类别不平衡',
                })

        # Precision/Recall差距大
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        if prec > 0 and rec > 0 and abs(prec - rec) > 0.2:
            deduction += 5
            if prec > rec:
                issues.append({'type': 'high_precision_low_recall', 'severity': 'low',
                               'message': '高准确率但低召回率 — 模型过于保守'})
                suggestions.append({
                    'priority': 2, 'param': 'decision_threshold', 'action': 'adjust',
                    'reason': '可降低分类阈值提升召回率',
                })

        return {'issues': issues, 'suggestions': suggestions, 'deduction': deduction}

    @staticmethod
    def _analyze_regression(metrics) -> dict:
        """回归特定分析"""
        issues = []
        suggestions = []
        deduction = 0

        r2 = metrics.get('r2', 0)
        mse = metrics.get('mse', 0)

        if r2 < 0:
            deduction += 20
            issues.append({
                'type': 'negative_r2', 'severity': 'high',
                'message': f'R²为负值 ({r2:.3f}) — 模型表现比均值预测还差',
            })
            suggestions.append({
                'priority': 1, 'param': 'target_standardization',
                'action': 'enable',
                'reason': '检查是否对target做了标准化 (StandardScaler)',
            })
            suggestions.append({
                'priority': 2, 'param': 'algorithm',
                'action': 'change',
                'suggested': 'gradient_boosting_regressor',
                'reason': '尝试GradientBoostingRegressor对回归任务效果更好',
            })

        if r2 > 0 and r2 < 0.40:
            deduction += 10
            issues.append({
                'type': 'low_r2', 'severity': 'medium',
                'message': f'R²偏低 ({r2:.3f})，模型解释力不足',
            })

        return {'issues': issues, 'suggestions': suggestions, 'deduction': deduction}

    @staticmethod
    def _generate_next_steps(health_score, issues, suggestions, hyperparams) -> list:
        """生成下一步操作步骤"""
        steps = []

        if health_score >= 85:
            steps = [
                "✅ 模型表现良好，可考虑部署上线",
                "📊 尝试模型对比，确认是否为最优选择",
                "🔧 微调超参数看能否进一步提升1-2%",
            ]
        elif health_score >= 65:
            steps = [
                f"🔧 优先调整: {suggestions[0]['param']} ({suggestions[0]['action']})" if suggestions else "🔧 尝试调整学习率或epochs",
                "📊 对比不同算法在同一数据集上的表现",
                "📈 检查是否需要更多训练数据",
            ]
        elif health_score >= 40:
            steps = [
                f"⚠️ 首先解决: {issues[0]['message']}" if issues else "⚠️ 检查数据预处理流程",
                "🔄 尝试更换算法 (如从sklearn切换到PyTorch深度学习)",
                "🔍 检查特征工程是否需要改进",
                "📊 使用超参数搜索 (GridSearchCV) 自动调优",
            ]
        else:
            steps = [
                "🔴 重新检查数据质量和预处理步骤",
                "🔄 尝试完全不同的算法和框架",
                "📊 确认目标变量是否正确",
                "🔍 考虑是否需要特征选择/降维",
            ]

        return steps
