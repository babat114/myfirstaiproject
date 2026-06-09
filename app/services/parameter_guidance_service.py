"""
============================================
训练参数诊断与调优引导服务 (v2 — Scientific Diagnostic Framework)
============================================

基于 2025-2026 ML 工程最佳实践全面重写:
  - Google Deep Learning Tuning Playbook 科学调参方法论
  - 诊断决策树 (Diagnosis Before Optimization)
  - 超参数三分类: Scientific / Nuisance / Fixed
  - 多维子评分体系 (绝对性能/过拟合风险/收敛质量/类别均衡/泛化能力)
  - 反模式检测 (Anti-Pattern Detection)
  - Learning Rate Finder 建议
  - 系统性调参顺序: LR → Batch Size → Regularization → Architecture
  - Optuna TPE Bayesian Optimization 集成建议

参考文献:
  - Google Developers: "A Scientific Approach to Improving Model Performance"
  - Kempner Institute: "Optimizing ML Workflows Workshop 2026"
  - Hyperopt → Optuna Migration Guide (Azure Databricks ML Runtime 16.4+)
"""
import json
import numpy as np
from typing import Optional
from app import logger


class ParameterGuidanceService:
    """训练参数科学诊断引擎 (v2)

    评分体系 (5维子评分 → 加权总分):
        ┌─────────────────────────────────────────────────────┐
        │ 1. 绝对性能 (Absolute Performance)    权重 35%     │
        │    - 基于 accuracy/F1/R²/silhouette 的实际值       │
        │    - 非线性映射: 从随机水平到SOTA分档              │
        │                                                     │
        │ 2. 过拟合风险 (Overfitting Risk)       权重 25%     │
        │    - train vs test gap 定量分析                     │
        │    - 严重→高→中→低 四级分类                        │
        │                                                     │
        │ 3. 收敛质量 (Convergence Quality)      权重 20%     │
        │    - Loss曲线震荡度 (CV变异系数)                    │
        │    - 收敛速度 (早期→晚期降幅)                       │
        │    - 梯度健康度 (如果可用)                          │
        │                                                     │
        │ 4. 类别均衡 (Class Balance)            权重 10%     │
        │    - Precision/Recall 差距                          │
        │    - F1与Accuracy的背离度                           │
        │                                                     │
        │ 5. 泛化能力 (Generalization)            权重 10%    │
        │    - 综合 train+test 表现                           │
        │    - 模型容量-数据量匹配度                          │
        └─────────────────────────────────────────────────────┘

    健康度分档 (科学校准):
        90-100  → 优秀 (Excellent) — 可部署
        75-89   → 良好 (Good) — 微调可提升1-3%
        60-74   → 一般 (Fair) — 存在可改进空间
        40-59   → 较差 (Poor) — 需要系统性优化
        25-39   → 很差 (Bad) — 需重新审视数据/特征
        <25     → 无效 (Ineffective) — 模型几乎无预测能力
    """

    # ── 超参数分类体系 ──
    # Scientific:  架构级参数，改变模型表达能力
    # Nuisance:   优化器参数，必须随Scientific变化而重新调优
    # Fixed:      已知良好默认值，当前实验轮次保持不变
    PARAM_CATEGORIES = {
        # Scientific (架构参数)
        'hidden_layers': 'scientific',
        'n_estimators': 'scientific',
        'max_depth': 'scientific',
        'kernel': 'scientific',
        'n_clusters': 'scientific',
        # Nuisance (优化器参数 — 必须在每次架构变化后重新调优)
        'learning_rate': 'nuisance',
        'batch_size': 'nuisance',
        'momentum': 'nuisance',
        'weight_decay': 'nuisance',
        'optimizer': 'nuisance',
        # Fixed (已知良好默认值)
        'early_stopping_patience': 'fixed',
        'val_size': 'fixed',
        'test_size': 'fixed',
        'random_state': 'fixed',
    }

    # ── 系统性调参顺序 (Google Playbook) ──
    TUNING_ORDER = [
        {'phase': 1, 'param': 'learning_rate', 'name': '学习率',
         'method': 'LR Finder (指数增长扫描)', 'reason': '单一最重要参数，决定收敛速度和最终性能'},
        {'phase': 2, 'param': 'batch_size', 'name': '批大小',
         'method': '线性缩放规则 (LR同步调整)', 'reason': '与LR耦合 — 翻倍batch需翻倍LR'},
        {'phase': 3, 'param': 'n_estimators', 'name': '树数量/层数',
         'method': '逐步增加至收益递减', 'reason': '控制模型容量，防止过拟合/欠拟合'},
        {'phase': 4, 'param': 'weight_decay', 'name': '权重衰减',
         'method': '对数尺度搜索 (1e-6 → 1e-2)', 'reason': '精细正则化，防止过拟合'},
        {'phase': 5, 'param': 'dropout', 'name': 'Dropout',
         'method': '线性扫描 (0.1 → 0.7)', 'reason': '最后一层防线 — 在LR和架构确定后微调'},
    ]

    # ── 反模式定义 ──
    ANTI_PATTERNS = {
        'tuning_on_test': {
            'name': '在测试集上调参',
            'severity': 'critical',
            'message': '检测到test指标被用于调参决策 — 这会导致乐观偏差和泛化失败',
            'fix': '使用嵌套交叉验证(Nested CV)或将测试集留到最后使用一次',
        },
        'fixing_optimizer_when_comparing_arch': {
            'name': '架构对比时固定优化器参数',
            'severity': 'high',
            'message': '比较不同架构时LR/optimizer被视为固定参数 — 可能导致不公平对比',
            'fix': '将LR/momentum等作为nuisance参数，为每种架构重新搜索最优LR',
        },
        'grid_search_high_dim': {
            'name': '高维Grid Search',
            'severity': 'medium',
            'message': '对>3个超参数使用Grid Search — 计算量指数爆炸，大部分搜索空间无效',
            'fix': '使用Optuna TPE Bayesian Optimization或至少Random Search',
        },
        'no_baseline': {
            'name': '缺少Baseline',
            'severity': 'medium',
            'message': '未建立默认参数Baseline就进行调优 — 无法量化改进幅度',
            'fix': '先用算法默认参数训练一次作为baseline，记录所有指标',
        },
    }

    # ==================================================================
    # 公共入口
    # ==================================================================

    @staticmethod
    def analyze_results(metrics_history: list, final_metrics: dict,
                        dataset_info: dict = None,
                        hyperparams: dict = None) -> dict:
        """全面科学诊断训练结果

        Returns:
            {
                'diagnosis': str,           # 综合诊断结论
                'health_score': float,       # 0-100 加权健康度
                'sub_scores': {             # 5维子评分
                    'absolute_perf': float,
                    'overfitting_risk': float,
                    'convergence_quality': float,
                    'class_balance': float,
                    'generalization': float,
                },
                'decision_path': [...],      # 诊断决策树路径
                'issues': [...],             # 检测到的问题清单
                'suggestions': [...],        # 按调参顺序排列的建议
                'next_steps': [...],         # 具体操作步骤
                'anti_patterns': [...],      # 检测到的反模式
                'tuning_roadmap': [...],     # 系统调参路线图
                'task_type': str,            # 任务类型
            }
        """
        metrics_history = metrics_history or []
        final_metrics = final_metrics or {}
        hyperparams = hyperparams or {}
        dataset_info = dataset_info or {}

        issues = []
        suggestions = []
        anti_patterns = []
        decision_path = []

        # 确定任务类型
        task_type = ParameterGuidanceService._detect_task_type(final_metrics, hyperparams)

        # ═══════════════════════════════════════════════════════════════
        # Step 1: 诊断决策树 — 结构化问题定位
        # ═══════════════════════════════════════════════════════════════
        dt_result = ParameterGuidanceService._diagnostic_decision_tree(
            metrics_history, final_metrics, task_type, hyperparams
        )
        issues.extend(dt_result['issues'])
        decision_path.extend(dt_result['path'])
        suggestions.extend(dt_result['suggestions'])

        # ═══════════════════════════════════════════════════════════════
        # Step 2: 5维子评分
        # ═══════════════════════════════════════════════════════════════
        sub_scores = {}

        # 2a. 绝对性能 (0-100)
        sub_scores['absolute_perf'] = ParameterGuidanceService._score_absolute_performance_v2(
            final_metrics, task_type
        )

        # 2b. 过拟合风险 (0-100, 100=无过拟合)
        sub_scores['overfitting_risk'] = ParameterGuidanceService._score_overfitting_v2(
            metrics_history, final_metrics, task_type
        )

        # 2c. 收敛质量 (0-100)
        sub_scores['convergence_quality'] = ParameterGuidanceService._score_convergence_v2(
            metrics_history, task_type
        )

        # 2d. 类别均衡 (0-100, 100=完美均衡)
        sub_scores['class_balance'] = ParameterGuidanceService._score_class_balance_v2(
            final_metrics, task_type
        )

        # 2e. 泛化能力 (0-100)
        sub_scores['generalization'] = ParameterGuidanceService._score_generalization_v2(
            metrics_history, final_metrics, task_type, hyperparams, dataset_info
        )

        # ═══════════════════════════════════════════════════════════════
        # Step 3: 加权总分
        # ═══════════════════════════════════════════════════════════════
        weights = {
            'absolute_perf': 0.35,
            'overfitting_risk': 0.25,
            'convergence_quality': 0.20,
            'class_balance': 0.10,
            'generalization': 0.10,
        }
        health_score = round(sum(
            sub_scores[k] * weights[k] for k in weights
        ), 1)

        # 边界校准 (防止过高/过低)
        health_score = max(3, min(99, health_score))

        # ═══════════════════════════════════════════════════════════════
        # Step 3.5: 子评分→问题映射 (补充决策树未覆盖的情况)
        # ═══════════════════════════════════════════════════════════════
        issues_from_scores = ParameterGuidanceService._issues_from_sub_scores(
            sub_scores, task_type, hyperparams
        )
        for iss in issues_from_scores:
            if not any(i.get('type') == iss['type'] for i in issues):
                issues.append(iss)

        # ═══════════════════════════════════════════════════════════════
        # Step 4: 反模式检测
        # ═══════════════════════════════════════════════════════════════
        anti_patterns = ParameterGuidanceService._detect_anti_patterns(
            metrics_history, final_metrics, hyperparams, task_type
        )

        # ═══════════════════════════════════════════════════════════════
        # Step 5: 系统性调参建议 (按科学顺序排列)
        # ═══════════════════════════════════════════════════════════════
        ordered_suggestions = ParameterGuidanceService._generate_scientific_suggestions(
            issues, sub_scores, task_type, hyperparams, health_score
        )

        # ═══════════════════════════════════════════════════════════════
        # Step 6: 综合诊断文本
        # ═══════════════════════════════════════════════════════════════
        diagnosis = ParameterGuidanceService._generate_diagnosis_text(
            health_score, sub_scores, issues, task_type
        )

        # ═══════════════════════════════════════════════════════════════
        # Step 7: 下一步操作 + 调参路线图
        # ═══════════════════════════════════════════════════════════════
        next_steps = ParameterGuidanceService._generate_scientific_next_steps(
            health_score, sub_scores, issues, ordered_suggestions, hyperparams
        )

        tuning_roadmap = ParameterGuidanceService._generate_tuning_roadmap(
            health_score, sub_scores, task_type, hyperparams
        )

        # 唯一化建议
        seen = set()
        unique_suggestions = []
        for s in ordered_suggestions:
            key = s.get('param', '') + s.get('action', '')
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        return {
            'diagnosis': diagnosis,
            'health_score': health_score,
            'sub_scores': sub_scores,
            'decision_path': decision_path,
            'issues': issues,
            'suggestions': unique_suggestions[:10],
            'next_steps': next_steps,
            'anti_patterns': anti_patterns,
            'tuning_roadmap': tuning_roadmap,
            'task_type': task_type,
            'weights': weights,
        }

    # ==================================================================
    # Step 1: 诊断决策树
    # ==================================================================

    @staticmethod
    def _diagnostic_decision_tree(history, final_metrics, task_type, hyperparams) -> dict:
        """结构化诊断决策树 — 按Google Playbook方法论

        决策流程:
            Q1: 训练Loss是否异常高? → YES: 检查LR/数据质量
            Q2: Val Loss >> Train Loss? → YES: 过拟合 → 增强正则化
            Q3: Train ≈ Val but 都低? → YES: 欠拟合 → 增加模型容量
            Q4: Loss震荡剧烈? → YES: LR过大或batch过小
            Q5: 收敛过慢? → YES: 提高LR或增加batch
            Q6: 指标健康? → YES: 正常 → 可微调
        """
        path = []
        issues = []
        suggestions = []

        n = len(history)
        has_enough_history = n >= 3

        # ── Q1: 训练Loss是否异常高? ──
        train_losses = [m.get('train_loss', m.get('loss')) for m in history
                        if m.get('train_loss') is not None or m.get('loss') is not None]

        if train_losses and len(train_losses) >= 3:
            final_loss = train_losses[-1]
            initial_loss = train_losses[0]

            # 极端高Loss (分类: >2.0, 回归: 需参考初始值)
            is_high_loss = False
            if task_type == 'classification':
                is_high_loss = final_loss > 2.0
            elif task_type == 'regression':
                is_high_loss = final_loss > initial_loss * 0.8 and final_loss > 10
            else:
                is_high_loss = final_loss > 5.0

            if is_high_loss:
                path.append({'node': 'Q1', 'question': '训练Loss是否异常高?',
                             'answer': 'YES', 'severity': 'high'})
                lr = hyperparams.get('learning_rate', 0.001)
                if lr > 0.01:
                    issues.append({
                        'type': 'high_initial_lr', 'severity': 'high',
                        'message': f'学习率({lr})可能过大导致Loss高居不下',
                        'category': 'nuisance',
                    })
                    suggestions.append({
                        'priority': 1, 'phase': 1,
                        'param': 'learning_rate', 'action': 'decrease',
                        'current': lr,
                        'suggested': round(lr / 5, 6),
                        'category': 'nuisance',
                        'reason': '学习率过高导致无法收敛 — 降低5x重试',
                        'method': 'LR Finder (指数增长扫描确定最佳LR范围)',
                    })
                else:
                    issues.append({
                        'type': 'high_loss_other', 'severity': 'high',
                        'message': '训练Loss异常高 — 可能数据质量问题或标签错误',
                    })
                    suggestions.append({
                        'priority': 1, 'phase': 0,
                        'param': 'data_quality', 'action': 'check',
                        'category': 'fixed',
                        'reason': '在调参之前，先排除数据质量问题: 检查缺失值/异常值/标签正确性',
                    })
            else:
                path.append({'node': 'Q1', 'question': '训练Loss是否异常高?',
                             'answer': 'NO — Loss正常范围内', 'severity': 'none'})

        # ── Q2: Val Loss >> Train Loss? (过拟合检测) ──
        overfit_info = ParameterGuidanceService._detect_overfitting_v2(
            history, final_metrics, task_type
        )
        if overfit_info['overfitting']:
            sev_label = '严重' if overfit_info['severity'] == 'high' else '轻微'
            path.append({'node': 'Q2', 'question': 'Val Loss >> Train Loss?',
                         'answer': f'YES — {sev_label}过拟合 (gap={overfit_info.get("gap", "?")})',
                         'severity': overfit_info['severity']})

            # 分级别建议
            if overfit_info['severity'] == 'high':
                suggestions.append({
                    'priority': 1, 'phase': 5,
                    'param': 'dropout', 'action': 'increase',
                    'current': hyperparams.get('dropout', 0.3),
                    'suggested': min(0.7, hyperparams.get('dropout', 0.3) + 0.3),
                    'category': 'nuisance',
                    'reason': '严重过拟合 — 大幅增加Dropout是最直接有效的正则化手段',
                })
                suggestions.append({
                    'priority': 2, 'phase': 4,
                    'param': 'weight_decay', 'action': 'increase',
                    'current': hyperparams.get('weight_decay', 1e-4),
                    'suggested': hyperparams.get('weight_decay', 1e-4) * 10,
                    'category': 'nuisance',
                    'reason': '配合增大weight_decay形成双重正则化防线',
                })
            else:
                suggestions.append({
                    'priority': 3, 'phase': 5,
                    'param': 'dropout', 'action': 'increase',
                    'current': hyperparams.get('dropout', 0.3),
                    'suggested': min(0.5, hyperparams.get('dropout', 0.3) + 0.15),
                    'category': 'nuisance',
                    'reason': '轻微过拟合 — 适度增加Dropout即可',
                })
        else:
            path.append({'node': 'Q2', 'question': 'Val Loss >> Train Loss?',
                         'answer': 'NO — Train/Val 差距正常', 'severity': 'none'})

        # ── Q3: 欠拟合检测 ──
        is_underfit = ParameterGuidanceService._detect_underfitting_v2(
            history, final_metrics, task_type
        )
        if is_underfit:
            path.append({'node': 'Q3', 'question': 'Train ≈ Val 但都很低?',
                         'answer': 'YES — 欠拟合', 'severity': 'medium'})
            suggestions.append({
                'priority': 2, 'phase': 3,
                'param': 'hidden_layers', 'action': 'increase',
                'current': hyperparams.get('hidden_layers', [128, 64]),
                'suggested': [256, 128, 64, 32],
                'category': 'scientific',
                'reason': '模型容量不足 — 增加网络深度/宽度提升表达能力',
            })
            suggestions.append({
                'priority': 1, 'phase': 3,
                'param': 'algorithm', 'action': 'change',
                'suggested': 'gradient_boosting',
                'category': 'scientific',
                'reason': '当前算法容量不足，建议切换到更强的集成方法(GB/RF)',
            })
        else:
            path.append({'node': 'Q3', 'question': 'Train ≈ Val 但都很低?',
                         'answer': 'NO', 'severity': 'none'})

        # ── Q4: Loss震荡 / 收敛速度? ──
        loss_analysis = ParameterGuidanceService._analyze_loss_curve_v2(history)
        if loss_analysis['trend'] == 'unknown' and len(history) < 3:
            # 单轮模型: 无足够epoch数据做loss曲线分析
            path.append({'node': 'Q4', 'question': 'Loss收敛情况?',
                         'answer': 'ⓘ 单轮训练, 无Loss曲线数据 — 跳过收敛分析',
                         'severity': 'none'})
        elif loss_analysis['oscillating']:
            path.append({'node': 'Q4', 'question': 'Loss是否剧烈震荡?',
                         'answer': f'YES — CV={loss_analysis["cv"]:.2f}',
                         'severity': 'medium'})
            current_lr = hyperparams.get('learning_rate', 0.001)
            current_bs = hyperparams.get('batch_size', 64)
            suggested_lr = round(max(current_lr / 5, 1e-6), 6)
            suggestions.append({
                'priority': 1, 'phase': 1,
                'param': 'learning_rate', 'action': 'decrease',
                'current': current_lr, 'suggested': suggested_lr,
                'category': 'nuisance',
                'reason': f'Loss震荡(CV={loss_analysis["cv"]:.1%}) → 学习率过大。'
                          f'标准修复: LR从{current_lr}降至{suggested_lr}',
            })
            if current_bs < 64:
                suggestions.append({
                    'priority': 3, 'phase': 2,
                    'param': 'batch_size', 'action': 'increase',
                    'current': current_bs, 'suggested': min(current_bs * 2, 256),
                    'category': 'nuisance',
                    'reason': '增大batch size可平滑梯度估计，减少震荡 (LR需同步缩放)',
                })
        elif loss_analysis['slow_convergence']:
            path.append({'node': 'Q4', 'question': 'Loss是否剧烈震荡?',
                         'answer': 'NO — 但收敛偏慢',
                         'severity': 'low'})
            current_lr = hyperparams.get('learning_rate', 0.001)
            suggested_lr = round(current_lr * 3, 6)
            suggestions.append({
                'priority': 4, 'phase': 1,
                'param': 'learning_rate', 'action': 'increase',
                'current': current_lr, 'suggested': suggested_lr,
                'category': 'nuisance',
                'reason': '收敛偏慢 — 可适当提高LR加速训练。运行LR Finder确定最佳范围',
            })
        else:
            path.append({'node': 'Q4', 'question': 'Loss是否剧烈震荡?',
                         'answer': 'NO — 收敛稳定', 'severity': 'none'})

        # ── Q5: LR-Batch Size 耦合检查 ──
        if has_enough_history:
            lr_bs_info = ParameterGuidanceService._check_lr_batch_coupling(
                history, hyperparams
            )
            if lr_bs_info['issue']:
                path.append({
                    'node': 'Q5', 'question': 'LR-Batch Size 耦合是否合理?',
                    'answer': f'问题: {lr_bs_info["message"]}',
                    'severity': 'medium',
                })
                suggestions.append({
                    'priority': 5, 'phase': 2,
                    'param': 'batch_size', 'action': lr_bs_info['bs_action'],
                    'suggested': lr_bs_info['suggested_bs'],
                    'category': 'nuisance',
                    'reason': lr_bs_info['reason'],
                })

        return {'path': path, 'issues': issues, 'suggestions': suggestions}

    # ==================================================================
    # Step 2a: 绝对性能评分 v2
    # ==================================================================

    @staticmethod
    def _score_absolute_performance_v2(final_metrics, task_type) -> float:
        """绝对性能评分 (非线性映射, 0-100)

        分档:
            95-100: SOTA级 — 接近理论上限
            85-94:  优秀 — 生产部署级别
            70-84:  良好 — 接近实用
            50-69:  一般 — 需要优化
            30-49:  较差 — 需要较大改进
            10-29:  很差 — 基本无效
            0-9:    无效 — 不如随机猜测
        """
        if task_type == 'classification':
            acc = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))
            f1 = final_metrics.get('test_f1_score', final_metrics.get('f1_score', 0))

            # Accuracy → 0-80 基础分 (非线性)
            if acc >= 0.98:    base = 78 + (acc - 0.98) / 0.02 * 2
            elif acc >= 0.95:  base = 68 + (acc - 0.95) / 0.03 * 10
            elif acc >= 0.88:  base = 55 + (acc - 0.88) / 0.07 * 13
            elif acc >= 0.80:  base = 40 + (acc - 0.80) / 0.08 * 15
            elif acc >= 0.70:  base = 25 + (acc - 0.70) / 0.10 * 15
            elif acc >= 0.55:  base = 10 + (acc - 0.55) / 0.15 * 15
            elif acc >= 0.40:  base = 3 + (acc - 0.40) / 0.15 * 7
            elif acc > 0:      base = max(0, acc * 7)
            else:              base = 0

            # F1 → ±15 调整
            if f1 > 0:
                if f1 >= 0.95:   bonus = 15
                elif f1 >= 0.85: bonus = 10
                elif f1 >= 0.75: bonus = 5
                elif f1 >= 0.60: bonus = 0
                elif f1 >= 0.45: bonus = -5
                elif f1 >= 0.30: bonus = -10
                else:            bonus = -15
            else:
                bonus = 0

            return round(max(0, min(100, base + bonus)), 1)

        elif task_type == 'regression':
            r2 = final_metrics.get('test_r2', final_metrics.get('r2', None))

            if r2 is not None:
                if r2 >= 0.95:   score = 92 + (r2 - 0.95) / 0.05 * 8
                elif r2 >= 0.85: score = 78 + (r2 - 0.85) / 0.10 * 14
                elif r2 >= 0.70: score = 60 + (r2 - 0.70) / 0.15 * 18
                elif r2 >= 0.50: score = 38 + (r2 - 0.50) / 0.20 * 22
                elif r2 >= 0.25: score = 18 + (r2 - 0.25) / 0.25 * 20
                elif r2 >= 0:    score = 5 + r2 / 0.25 * 13
                else:            score = max(0, 3 + r2 * 3)
                return round(max(0, min(100, score)), 1)
            return 20.0  # 无R²指标

        elif task_type == 'clustering':
            sil = final_metrics.get('test_silhouette_score',
                                    final_metrics.get('silhouette_score', None))
            if sil is not None:
                if sil >= 0.70:  score = 88 + (sil - 0.70) / 0.30 * 12
                elif sil >= 0.50: score = 65 + (sil - 0.50) / 0.20 * 23
                elif sil >= 0.30: score = 38 + (sil - 0.30) / 0.20 * 27
                elif sil >= 0.10: score = 15 + (sil - 0.10) / 0.20 * 23
                elif sil >= 0:    score = 5 + sil / 0.10 * 10
                else:             score = max(0, 2 + sil * 3)
                return round(max(0, min(100, score)), 1)
            return 20.0

        return 20.0

    # ==================================================================
    # Step 2b: 过拟合风险评分 v2
    # ==================================================================

    @staticmethod
    def _score_overfitting_v2(history, final_metrics, task_type) -> float:
        """过拟合风险评分 (100 = 完全无过拟合, 0 = 严重过拟合)"""
        info = ParameterGuidanceService._detect_overfitting_v2(
            history, final_metrics, task_type
        )
        if not info['overfitting']:
            return 100.0
        if info['severity'] == 'critical':
            return 10.0
        elif info['severity'] == 'high':
            return 35.0
        elif info['severity'] == 'medium':
            return 55.0
        else:
            return 75.0

    @staticmethod
    def _detect_overfitting_v2(history, final_metrics, task_type) -> dict:
        """增强过拟合检测 — 多信号融合"""
        if len(history) < 1:
            return {'overfitting': False, 'severity': 'none', 'gap': 0}

        n_recent = max(1, len(history) // 3)
        recent = history[-n_recent:]

        if task_type == 'regression':
            train_key, test_key = 'train_r2', 'test_r2'
            train_vals = [m.get(train_key, m.get('r2')) for m in recent
                          if m.get(train_key) is not None or m.get('r2') is not None]
            test_val = final_metrics.get('test_r2', final_metrics.get('r2'))
        elif task_type == 'clustering':
            return {'overfitting': False, 'severity': 'none', 'gap': 0}
        else:
            train_key, test_key = 'train_accuracy', 'test_accuracy'
            train_vals = [m.get(train_key, m.get('accuracy')) for m in recent
                          if m.get(train_key) is not None or m.get('accuracy') is not None]
            test_val = final_metrics.get('test_accuracy', final_metrics.get('accuracy'))

        if not train_vals or test_val is None:
            return {'overfitting': False, 'severity': 'none', 'gap': 0}

        avg_train = float(np.mean(train_vals))
        gap = avg_train - test_val

        if task_type == 'regression':
            if gap > 0.30:   return {'overfitting': True, 'severity': 'critical', 'gap': round(gap, 4)}
            elif gap > 0.20: return {'overfitting': True, 'severity': 'high', 'gap': round(gap, 4)}
            elif gap > 0.10: return {'overfitting': True, 'severity': 'medium', 'gap': round(gap, 4)}
            elif gap > 0.05: return {'overfitting': True, 'severity': 'low', 'gap': round(gap, 4)}
        else:
            if gap > 0.20:   return {'overfitting': True, 'severity': 'critical', 'gap': round(gap, 4)}
            elif gap > 0.12: return {'overfitting': True, 'severity': 'high', 'gap': round(gap, 4)}
            elif gap > 0.06: return {'overfitting': True, 'severity': 'medium', 'gap': round(gap, 4)}
            elif gap > 0.03: return {'overfitting': True, 'severity': 'low', 'gap': round(gap, 4)}

        return {'overfitting': False, 'severity': 'none', 'gap': round(gap, 4)}

    # ==================================================================
    # Step 2c: 收敛质量评分 v2
    # ==================================================================

    @staticmethod
    def _score_convergence_v2(history, task_type) -> float:
        """收敛质量评分 (100 = 完美收敛, 0 = 完全未收敛)

        对于单轮模型(sklearn 1-epoch): 无法分析Loss曲线, 基于最终指标推算
        """
        n = len(history)
        if n < 3:
            # 单轮模型: 无法做loss曲线分析, 给中性偏高分数(不扣分也不加分)
            # 收敛质量默认为不可评估而非差 — 避免误导低分
            return 70.0

        analysis = ParameterGuidanceService._analyze_loss_curve_v2(history)

        # 基础分 + 扣分项
        score = 80.0

        if analysis['oscillating']:
            cv = analysis['cv']
            if cv > 0.50:     score -= 40
            elif cv > 0.35:   score -= 30
            elif cv > 0.25:   score -= 20
            else:             score -= 10

        if analysis['slow_convergence']:
            score -= 15

        if analysis['diverging']:
            score -= 50  # 发散是最严重的信号

        if analysis['converged_well']:
            score = min(100, score + 20)

        return round(max(0, min(100, score)), 1)

    @staticmethod
    def _analyze_loss_curve_v2(history) -> dict:
        """增强Loss曲线分析 — 多维度信号"""
        if len(history) < 3:
            return {'oscillating': False, 'slow_convergence': False,
                    'converged_well': False, 'diverging': False,
                    'cv': 0, 'trend': 'unknown'}

        losses = [m.get('loss', m.get('train_loss')) for m in history
                  if m.get('loss') is not None or m.get('train_loss') is not None]
        if len(losses) < 3:
            return {'oscillating': False, 'slow_convergence': False,
                    'converged_well': False, 'diverging': False,
                    'cv': 0, 'trend': 'unknown'}

        avg_loss = np.mean(losses)
        if avg_loss <= 0:
            return {'oscillating': False, 'slow_convergence': False,
                    'converged_well': False, 'diverging': False,
                    'cv': 0, 'trend': 'unknown'}

        # 变异系数 (CV = std/mean)
        cv = float(np.std(losses) / avg_loss)

        # 趋势检测 (线性回归斜率)
        x = np.arange(len(losses))
        slope = float(np.polyfit(x, losses, 1)[0])

        # 发散检测: 后期Loss > 初期Loss * 1.5
        early_mean = float(np.mean(losses[:min(3, len(losses))]))
        late_mean = float(np.mean(losses[-min(3, len(losses)):]))
        diverging = late_mean > early_mean * 1.5

        # 收敛检测
        converged_well = (late_mean < early_mean * 0.3) and (cv < 0.12)
        slow = late_mean > early_mean * 0.7 and not diverging

        return {
            'oscillating': cv > 0.25,
            'slow_convergence': slow,
            'converged_well': converged_well,
            'diverging': diverging,
            'cv': round(cv, 4),
            'trend': 'diverging' if diverging else
                     'converging' if slope < -0.01 else
                     'flat' if abs(slope) < 0.01 else 'increasing',
            'slope': round(float(slope), 6),
            'early_mean': round(early_mean, 4),
            'late_mean': round(late_mean, 4),
        }

    # ==================================================================
    # Step 2d: 类别均衡评分 v2
    # ==================================================================

    @staticmethod
    def _score_class_balance_v2(final_metrics, task_type) -> float:
        """类别均衡评分 (100 = 完美均衡, 仅适用于分类任务)"""
        if task_type != 'classification':
            return 100.0

        prec = final_metrics.get('test_precision', final_metrics.get('precision', 0))
        rec = final_metrics.get('test_recall', final_metrics.get('recall', 0))
        f1 = final_metrics.get('test_f1_score', final_metrics.get('f1_score', 0))
        acc = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))

        score = 100.0

        # P-R 差距过大
        if prec > 0 and rec > 0:
            pr_gap = abs(prec - rec)
            if pr_gap > 0.40:    score -= 35
            elif pr_gap > 0.25:  score -= 20
            elif pr_gap > 0.15:  score -= 10
            elif pr_gap > 0.08:  score -= 5

        # F1 远低于 Accuracy (类别不平衡的标志)
        if acc > 0 and f1 > 0:
            acc_f1_gap = acc - f1
            if acc_f1_gap > 0.30:   score -= 25
            elif acc_f1_gap > 0.15: score -= 12
            elif acc_f1_gap > 0.08: score -= 5

        # 极端P或R
        if prec > 0 and prec < 0.30:  score -= 15
        if rec > 0 and rec < 0.30:   score -= 15

        return round(max(0, min(100, score)), 1)

    # ==================================================================
    # Step 2e: 泛化能力评分 v2
    # ==================================================================

    @staticmethod
    def _score_generalization_v2(history, final_metrics, task_type,
                                  hyperparams, dataset_info) -> float:
        """泛化能力评分 (100 = 完美泛化)"""
        score = 80.0  # 基础分

        # Train-Test gap 信号
        n = len(history)
        if n >= 2:
            if task_type == 'classification':
                train_vals = [m.get('train_accuracy', m.get('accuracy')) for m in history
                              if m.get('train_accuracy') is not None or m.get('accuracy') is not None]
                test_val = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))
                if train_vals:
                    avg_train = float(np.mean(train_vals))
                    gap = avg_train - test_val
                    if gap > 0.15:     score -= 30
                    elif gap > 0.08:   score -= 15
                    elif gap > 0.03:   score -= 5
            elif task_type == 'regression':
                train_r2s = [m.get('train_r2', m.get('r2')) for m in history
                             if m.get('train_r2') is not None or m.get('r2') is not None]
                test_r2 = final_metrics.get('test_r2', final_metrics.get('r2', 0))
                if train_r2s:
                    avg_train = float(np.mean(train_r2s))
                    gap = avg_train - test_r2
                    if gap > 0.20:    score -= 30
                    elif gap > 0.10:  score -= 15
                    elif gap > 0.05:  score -= 5

        # 模型容量-数据量匹配 (如果可用)
        n_samples = dataset_info.get('n_samples', 0) if dataset_info else 0
        hl = hyperparams.get('hidden_layers', [])
        if hl and n_samples > 0:
            total_params_est = sum(hl) * 100  # 粗略估计
            ratio = n_samples / max(total_params_est, 1)
            if ratio < 1:     score -= 15  # 参数多于样本 → 高风险过拟合
            elif ratio < 5:   score -= 5
            elif ratio > 50:  score += 5   # 充足数据

        return round(max(0, min(100, score)), 1)

    # ==================================================================
    # Step 3.5: 子评分→问题映射
    # ==================================================================

    @staticmethod
    def _issues_from_sub_scores(sub_scores, task_type, hyperparams) -> list:
        """从子评分反推问题 — 确保低分维度一定有对应的问题描述

        v2.1: 降低阈值使诊断更敏感 — 之前的阈值过高导致大部分模型无任何issue
        """
        issues = []
        thresholds = {
            'absolute_perf': (75, '绝对性能不足', '模型预测能力未达到可用水平'),
            'overfitting_risk': (70, '存在过拟合风险', '训练/验证集指标差距偏大'),
            'convergence_quality': (70, '收敛质量偏低', 'Loss曲线收敛不理想或无法评估(单轮模型)'),
            'class_balance': (75, '类别不均衡影响', 'Precision/Recall差距偏大'),
            'generalization': (70, '泛化能力不足', '模型在新数据上表现可能不稳定'),
        }

        for dim, (threshold, title, desc) in thresholds.items():
            score = sub_scores.get(dim, 100)
            if score < threshold:
                severity = 'high' if score < 40 else 'medium' if score < 55 else 'low'
                issues.append({
                    'type': f'low_{dim}',
                    'severity': severity,
                    'message': f'{title}: {desc} (评分 {score:.0f}/100)',
                    'category': 'scientific' if dim == 'absolute_perf' else 'nuisance',
                })
        return issues

    # ==================================================================
    # Step 4: 反模式检测
    # ==================================================================

    @staticmethod
    def _detect_anti_patterns(history, final_metrics, hyperparams, task_type) -> list:
        """检测常见反模式"""
        found = []

        # 反模式1: 无有效Baseline — v2.1 修复: 已有指标即为有效baseline
        has_meaningful_metrics = bool(
            final_metrics.get('accuracy') or final_metrics.get('f1_score') or
            final_metrics.get('r2') is not None or
            final_metrics.get('silhouette_score') or
            final_metrics.get('test_accuracy') or final_metrics.get('test_f1_score') or
            final_metrics.get('test_r2') is not None or
            final_metrics.get('test_silhouette_score')
        )
        if not hyperparams.get('tuned') and not hyperparams.get('baseline_score'):
            if not has_meaningful_metrics and len(history) < 2:
                found.append({
                    'type': 'no_baseline',
                    'name': '缺少有效Baseline',
                    'severity': 'medium',
                    'message': '未检测到有效指标 — 模型可能未成功训练或指标为空',
                    'fix': '检查训练日志，确认模型成功完成训练并产生了指标',
                })

        # 反模式2: GridSearch高维
        if hyperparams.get('tuning_method') == 'grid' and hyperparams.get('tuned'):
            hp_count = len([k for k in hyperparams if k not in
                           ('task_type', 'algorithm', 'target_column', 'tuned',
                            'tuning_method', 'best_cv_score', 'tuning_cv_folds')])
            if hp_count > 3:
                found.append({
                    'type': 'grid_search_high_dim',
                    'name': '高维Grid Search',
                    'severity': 'medium',
                    'message': f'对{hp_count}个超参数使用Grid Search — 推荐切换到Optuna TPE Bayesian Optimization',
                    'fix': '使用Optuna进行贝叶斯优化，或至少切换到Random Search',
                })

        # 反模式3: 测试集泄露风险
        if final_metrics:
            metric_keys = set(final_metrics.keys())
            suspicious = {'test_accuracy', 'test_f1', 'test_r2'} & metric_keys
            if suspicious and len(history) > 10:
                # 多轮调整后还在报告test指标 → 可能存在test调参
                found.append({
                    'type': 'potential_test_leak',
                    'name': '潜在的测试集过拟合',
                    'severity': 'low',
                    'message': '警告: 经过多轮训练后仍报告test指标 — 确保test集仅用于最终评估',
                    'fix': '采用嵌套CV: 外层CV评估泛化性能，内层CV选择超参数',
                })

        return found

    # ==================================================================
    # Step 5: 系统性调参建议
    # ==================================================================

    @staticmethod
    def _generate_scientific_suggestions(issues, sub_scores, task_type,
                                          hyperparams, health_score) -> list:
        """按科学调参顺序生成建议 (Phase 1→5)

        v2.1: 降低触发阈值 + 算法感知 — 不再对树模型建议dropout/batch_size
        """
        suggestions = []
        algorithm = hyperparams.get('algorithm', '')
        algo_lower = algorithm.lower()

        # 算法分类 (决定哪些建议有意义)
        _TREE_ALGOS = {'random_forest', 'random_forest_regressor', 'gradient_boosting',
                       'gradient_boosting_regressor', 'decision_tree'}
        _LINEAR_ALGOS = {'logistic_regression', 'linear_regression', 'ridge', 'svm', 'svr'}
        _KNN_ALGOS = {'knn', 'knn_regressor'}
        _CLUSTER_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}
        _DL_ALGOS = {'mlp', 'transformer_bert'}
        is_tree = algo_lower in _TREE_ALGOS
        is_cluster = algo_lower in _CLUSTER_ALGOS or task_type == 'clustering'
        is_dl = algo_lower in _DL_ALGOS

        # 收敛质量差 → Phase 1: LR (仅对DL/GB有意义)
        if sub_scores.get('convergence_quality', 100) < 70 and not is_cluster:
            current_lr = hyperparams.get('learning_rate', 0.001)
            if is_tree and algo_lower.startswith('gradient'):
                # GBDT: 调整learning_rate
                suggestions.append({
                    'priority': 1, 'phase': 1,
                    'param': 'learning_rate', 'action': 'decrease',
                    'current': current_lr,
                    'suggested': round(max(current_lr / 3, 1e-6), 6),
                    'category': 'nuisance',
                    'reason': f'收敛质量评分偏低({sub_scores["convergence_quality"]:.0f}/100) — '
                              f'降低GBDT学习率+增加n_estimators补偿',
                    'method': 'GBDT: 学习率减半 → n_estimators加倍',
                })
            elif is_dl:
                suggestions.append({
                    'priority': 1, 'phase': 1,
                    'param': 'learning_rate', 'action': 'decrease',
                    'current': current_lr,
                    'suggested': round(max(current_lr / 5, 1e-6), 6),
                    'category': 'nuisance',
                    'reason': f'收敛质量评分偏低({sub_scores["convergence_quality"]:.0f}/100) — '
                              f'优先调整学习率 (Phase 1: 最重要的参数)',
                    'method': '运行LR Finder确定最佳LR范围',
                })

        # 过拟合风险高 → 正则化 (不同算法有不同正则化手段)
        if sub_scores.get('overfitting_risk', 100) < 75:
            if is_tree:
                suggestions.append({
                    'priority': 2, 'phase': 3,
                    'param': 'max_depth', 'action': 'decrease',
                    'current': hyperparams.get('max_depth', 'None'),
                    'suggested': hyperparams.get('max_depth') or 10,
                    'category': 'scientific',
                    'reason': f'过拟合风险评分({sub_scores["overfitting_risk"]:.0f}/100) — '
                              f'树模型: 限制max_depth是最有效的正则化手段',
                })
                suggestions.append({
                    'priority': 3, 'phase': 3,
                    'param': 'min_samples_split', 'action': 'increase',
                    'current': hyperparams.get('min_samples_split', 2),
                    'suggested': min(hyperparams.get('min_samples_split', 2) * 3, 20),
                    'category': 'scientific',
                    'reason': '增大min_samples_split进一步防止过拟合',
                })
            elif is_dl:
                suggestions.append({
                    'priority': 2, 'phase': 4,
                    'param': 'weight_decay', 'action': 'increase',
                    'current': hyperparams.get('weight_decay', 1e-4),
                    'suggested': hyperparams.get('weight_decay', 1e-4) * 5,
                    'category': 'nuisance',
                    'reason': f'过拟合风险评分({sub_scores["overfitting_risk"]:.0f}/100) — '
                              f'Phase 4: 增强weight_decay正则化',
                })
                suggestions.append({
                    'priority': 3, 'phase': 5,
                    'param': 'dropout', 'action': 'increase',
                    'current': hyperparams.get('dropout', 0.3),
                    'suggested': min(0.7, hyperparams.get('dropout', 0.3) + 0.2),
                    'category': 'nuisance',
                    'reason': '配合增加dropout形成双重正则化防线',
                })
            elif not is_cluster:
                # SVM/线性模型: 增强C/alpha
                suggestions.append({
                    'priority': 2, 'phase': 4,
                    'param': 'C' if 'svm' in algo_lower else 'alpha',
                    'action': 'decrease',
                    'current': hyperparams.get('C', hyperparams.get('alpha', 1.0)),
                    'suggested': round(hyperparams.get('C', hyperparams.get('alpha', 1.0)) / 5, 4),
                    'category': 'nuisance',
                    'reason': f'过拟合风险评分({sub_scores["overfitting_risk"]:.0f}/100) — '
                              f'增强L2正则化 (减小C/alpha = 更强正则化)',
                })

        # 类别不均衡 → class_weight
        if sub_scores.get('class_balance', 100) < 80 and task_type == 'classification':
            suggestions.append({
                'priority': 3, 'phase': 1,
                'param': 'class_weight', 'action': 'enable',
                'suggested': 'balanced',
                'category': 'scientific',
                'reason': f'类别均衡评分偏低({sub_scores["class_balance"]:.0f}/100) — '
                          f'启用class_weight="balanced"处理类别不平衡',
            })

        # 绝对性能低 → 提升模型容量
        if sub_scores.get('absolute_perf', 100) < 70:
            if is_tree:
                suggestions.append({
                    'priority': 4, 'phase': 3,
                    'param': 'n_estimators', 'action': 'increase',
                    'current': hyperparams.get('n_estimators', 200),
                    'suggested': min(hyperparams.get('n_estimators', 200) * 2, 500),
                    'category': 'scientific',
                    'reason': f'绝对性能评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                              f'增加树数量提升模型表达能力',
                })
            elif is_dl:
                hl = hyperparams.get('hidden_layers', [128, 64, 32])
                wider = [min(h * 2, 1024) for h in hl]
                suggestions.append({
                    'priority': 4, 'phase': 3,
                    'param': 'hidden_layers', 'action': 'widen',
                    'current': hl,
                    'suggested': wider,
                    'category': 'scientific',
                    'reason': f'绝对性能评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                              f'加宽网络提升模型容量',
                })
            elif is_cluster:
                suggestions.append({
                    'priority': 4, 'phase': 3,
                    'param': 'algorithm', 'action': 'change',
                    'suggested': 'kmeans' if algo_lower != 'kmeans' else 'agglomerative',
                    'category': 'scientific',
                    'reason': f'聚类质量评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                              f'尝试不同聚类算法。K-Means适合球形簇，DBSCAN适合任意形状',
                })
            elif algo_lower in _LINEAR_ALGOS:
                suggestions.append({
                    'priority': 4, 'phase': 3,
                    'param': 'algorithm', 'action': 'change',
                    'suggested': 'random_forest',
                    'category': 'scientific',
                    'reason': f'绝对性能评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                              f'线性模型容量有限，建议升级到RandomForest/GBDT',
                })

        # 泛化能力低
        if sub_scores.get('generalization', 100) < 75:
            n_samples = hyperparams.get('_n_samples', 0)
            if n_samples < 1000:
                suggestions.append({
                    'priority': 5, 'phase': 5,
                    'param': 'test_size', 'action': 'increase',
                    'current': hyperparams.get('test_size', 0.2),
                    'suggested': 0.3,
                    'category': 'fixed',
                    'reason': f'泛化能力评分偏低({sub_scores["generalization"]:.0f}/100) + '
                              f'小样本({n_samples}) — 增加测试集比例以获得更可靠的泛化估计',
                })

        return suggestions

    # ==================================================================
    # Step 6: 综合诊断文本
    # ==================================================================

    @staticmethod
    def _generate_diagnosis_text(health_score, sub_scores, issues, task_type) -> str:
        """生成综合诊断文本 — v2.1 算法感知: 针对不同情况给出不同建议"""
        worst_dim = min(sub_scores, key=sub_scores.get) if sub_scores else None
        worst_score = sub_scores.get(worst_dim, 0) if worst_dim else 0

        dim_names = {
            'absolute_perf': '绝对性能',
            'overfitting_risk': '过拟合风险',
            'convergence_quality': '收敛质量',
            'class_balance': '类别均衡',
            'generalization': '泛化能力',
        }

        # 识别弱项 (评分 < 80 的维度)
        weak_dims = [(dim_names.get(k, k), v) for k, v in sub_scores.items()
                     if v < 80 and k != 'overfitting_risk']
        weak_dims_str = '、'.join(f'{name}({score:.0f}/100)'
                                  for name, score in weak_dims[:3]) if weak_dims else '无'

        task_cn = {'classification': '分类', 'regression': '回归',
                   'clustering': '聚类'}.get(task_type, task_type)

        if health_score >= 90:
            return (f"[{task_cn}任务] 模型表现卓越 (健康度 {health_score:.0f}/100)。"
                    f"各项指标优秀，已达到生产部署标准。")
        elif health_score >= 75:
            return (f"[{task_cn}任务] 模型表现良好 (健康度 {health_score:.0f}/100)。"
                    f"弱项维度: {weak_dims_str}。"
                    f"按下方路线图微调可预期提升1-3个百分点。")
        elif health_score >= 60:
            return (f"[{task_cn}任务] 模型表现一般 (健康度 {health_score:.0f}/100)。"
                    f"弱项维度: {weak_dims_str}。"
                    f"检测到 {len(issues)} 个可改进问题，建议按下方路线图逐步优化。")
        elif health_score >= 40:
            return (f"[{task_cn}任务] 模型表现偏差 (健康度 {health_score:.0f}/100)。"
                    f"核心问题: {dim_names.get(worst_dim, '?')} 仅 {worst_score:.0f}/100。"
                    f"弱项维度: {weak_dims_str}。需要系统性优化。")
        elif health_score >= 25:
            return (f"[{task_cn}任务] 模型表现很差 (健康度 {health_score:.0f}/100)，"
                    f"基本不具备实用价值。建议: "
                    f"1) 重新检查数据质量 2) 使用RF/GB建立baseline 3) 特征选择去噪。")
        else:
            return (f"[{task_cn}任务] 模型几乎无效 (健康度 {health_score:.0f}/100)。"
                    f"请确认: 1) 目标列设置正确? 2) 数据包含有效标签? "
                    f"3) 数据格式无问题? 建议从LogisticRegression/Ridge baseline开始。")

    # ==================================================================
    # Step 7: 下一步操作 + 调参路线图
    # ==================================================================

    @staticmethod
    def _generate_scientific_next_steps(health_score, sub_scores, issues,
                                         suggestions, hyperparams) -> list:
        """生成科学调参步骤清单 — v2.1 降低触发阈值 + 算法感知"""
        steps = []
        algorithm = hyperparams.get('algorithm', '')
        algo_lower = algorithm.lower()
        is_cluster = algo_lower in {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}

        cq = sub_scores.get('convergence_quality', 100)
        of = sub_scores.get('overfitting_risk', 100)
        ap = sub_scores.get('absolute_perf', 100)

        # 第1步: 根据问题严重程度生成针对性建议
        if cq < 75 and not is_cluster:
            steps.append({
                'step': 1, 'phase': 'Phase 1',
                'action': '调整学习率 (最重要参数)',
                'detail': f'收敛质量 {cq:.0f}/100 → 运行LR Finder找到最佳LR范围',
                'icon': '🔍',
            })

        if of < 75 and not is_cluster:
            steps.append({
                'step': 2, 'phase': 'Phase 4-5',
                'action': '增强正则化',
                'detail': f'过拟合风险 {of:.0f}/100 → 依次调整正则化参数' +
                          (' (weight_decay → dropout)' if algo_lower in ('mlp',) else
                           ' (max_depth → min_samples_split)' if algo_lower.startswith('random_forest') else
                           ' (C/alpha → 简化模型)'),
                'icon': '🛡️',
            })

        if ap < 75 or len(issues) > 0:
            steps.append({
                'step': 3 if steps else 1, 'phase': 'Phase 3',
                'action': '提升模型容量/切换算法',
                'detail': f'绝对性能 {ap:.0f}/100 → ' +
                          ('增加n_estimators或切换GBDT' if algo_lower in ('random_forest', 'random_forest_regressor') else
                           '尝试更深的网络' if algo_lower in ('mlp',) else
                           '尝试不同聚类算法' if is_cluster else
                           '考虑升级到集成方法(RandomForest/GBDT)'),
                'icon': '📈',
            })

        # Optuna建议 — 有多个issue时推荐
        if health_score < 85 and len(issues) >= 1:
            steps.append({
                'step': 4 if steps else 1, 'phase': '进阶',
                'action': '使用Optuna自动搜索',
                'detail': '对确定的2-3个关键超参数运行Optuna TPE搜索 (50-100 trials)',
                'icon': '🤖',
            })

        # 良好模型 — 给出部署建议而非优化建议
        if health_score >= 85:
            steps.append({
                'step': len(steps) + 1, 'phase': '部署',
                'action': '模型已达到生产标准',
                'detail': '可进行: 模型压缩/量化 → 部署上线 → 持续监控drift',
                'icon': '✅',
            })

        return steps

    @staticmethod
    def _generate_tuning_roadmap(health_score, sub_scores, task_type, hyperparams) -> list:
        """生成调参路线图 (Phase 1→5) — v2.1 算法感知: 过滤不相关的Phase

        树模型(RF/GBDT): 过滤dropout/batch_size — 对bagging/boosting无意义
        聚类模型: 替换为聚类专用路线图
        线性模型(SVM/LR): 过滤dropout/n_estimators
        """
        algorithm = hyperparams.get('algorithm', '')
        algo_lower = algorithm.lower()

        # 算法感知: 定义各Phase对哪些算法有意义
        _DL_ONLY = {'mlp', 'transformer_bert'}   # dropout/batch_size/lr仅对DL有意义
        _TREE_ONLY = {'random_forest', 'random_forest_regressor',
                      'gradient_boosting', 'gradient_boosting_regressor',
                      'decision_tree'}            # n_estimators/max_depth仅对树模型有意义
        _CLUSTER_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}

        is_dl = algo_lower in _DL_ONLY
        is_tree = algo_lower in _TREE_ONLY
        is_cluster = algo_lower in _CLUSTER_ALGOS or task_type == 'clustering'

        # ── 聚类专属路线图 ──
        if is_cluster:
            return [
                {'phase': 1, 'param': 'n_clusters', 'name': '簇数选择',
                 'method': 'Elbow Method + Silhouette Analysis',
                 'reason': '确定最佳簇数是聚类最重要的决策',
                 'urgency': 'high' if sub_scores.get('absolute_perf', 100) < 70 else 'medium',
                 'note': f'当前silhouette仅{sub_scores.get("absolute_perf", 0):.0f}/100 — 尝试不同k值' if sub_scores.get('absolute_perf', 100) < 70 else '通过Elbow曲线找到拐点'},
                {'phase': 2, 'param': 'algorithm', 'name': '聚类算法选择',
                 'method': '对比 K-Means vs DBSCAN vs Agglomerative',
                 'reason': '不同算法假设不同的簇形状 — K-Means球形/DBSCAN任意形/层次树形',
                 'urgency': 'medium',
                 'note': f'当前{algorithm}, 建议对比其他算法'},
                {'phase': 3, 'param': 'init', 'name': '初始化方法',
                 'method': 'k-means++ vs random',
                 'reason': 'k-means++ 提供更好的初始质心，通常收敛到更优解',
                 'urgency': 'low' if algo_lower != 'kmeans' else 'medium'},
                {'phase': 4, 'param': 'max_iter', 'name': '最大迭代次数',
                 'method': '逐步增加至收敛稳定 (100 → 500)',
                 'reason': '复杂数据集需要更多迭代才能收敛',
                 'urgency': 'low'},
            ]

        roadmap = []
        cq = sub_scores.get('convergence_quality', 100)
        of = sub_scores.get('overfitting_risk', 100)
        ap = sub_scores.get('absolute_perf', 100)

        for phase_info in ParameterGuidanceService.TUNING_ORDER:
            phase = phase_info.copy()
            param = phase['param']

            # ── 算法感知过滤 ──
            if param in ('dropout',) and not is_dl and not is_tree:
                # Dropout 仅对 DL + GBDT(sub_sample可类比) 有意义
                if algo_lower not in ('gradient_boosting', 'gradient_boosting_regressor'):
                    continue
                else:
                    phase['name'] = 'Subsampling'
                    phase['method'] = '调整 subsample (0.6 → 1.0)'
                    phase['reason'] = 'GBDT的随机采样类似dropout — 降低subsample增加随机性防过拟合'
            if param == 'batch_size' and not is_dl:
                continue  # 树模型/线性模型没有batch_size概念
            if param == 'weight_decay' and not is_dl:
                if algo_lower in ('svm', 'svr'):
                    phase['name'] = 'C (正则化强度)'
                    phase['method'] = '对数尺度搜索 (0.01 → 100)'
                    phase['reason'] = 'SVM的C参数控制正则化 — 越小=越强正则化'
                elif algo_lower in ('logistic_regression', 'ridge', 'linear_regression'):
                    phase['name'] = 'C/alpha (正则化强度)'
                    phase['method'] = '对数尺度搜索'
                    phase['reason'] = '线性模型调整正则化强度控制过拟合'
                elif is_tree:
                    phase['name'] = 'min_samples_split + max_depth'
                    phase['method'] = '网格搜索: max_depth(3→20) x min_samples_split(2→20)'
                    phase['reason'] = '树模型通过限制分裂控制过拟合 — 比weight_decay更有效'
                else:
                    continue
            if param == 'n_estimators' and not is_tree and not is_dl:
                # n_estimators对线性模型没有意义
                continue

            # 根据当前状态标记优先级
            if param == 'learning_rate' and cq < 75:
                phase['urgency'] = 'high'
                phase['note'] = f'收敛质量仅{cq:.0f}/100 — 立即调整'
            elif param in ('weight_decay', 'dropout') and of < 75:
                phase['urgency'] = 'high'
                phase['note'] = f'过拟合风险评分{of:.0f}/100 — 需要增强正则化'
            elif param == 'n_estimators' and ap < 70:
                phase['urgency'] = 'high'
                phase['note'] = f'绝对性能仅{ap:.0f}/100 — 增加模型容量'
            elif health_score >= 85:
                phase['urgency'] = 'low'
                phase['note'] = '当前表现良好，可选微调'
            else:
                phase['urgency'] = 'medium'

            current_val = hyperparams.get(phase['param'])
            if current_val is not None:
                phase['current'] = current_val

            roadmap.append(phase)

        return roadmap

    # ==================================================================
    # 辅助工具
    # ==================================================================

    @staticmethod
    def _check_lr_batch_coupling(history, hyperparams) -> dict:
        """LR-Batch Size 耦合检查"""
        lr = hyperparams.get('learning_rate', 0.001)
        bs = hyperparams.get('batch_size', 64)

        # 线性缩放规则: 翻倍batch → 翻倍LR
        # 标准参考点: bs=256, lr=0.1 (ImageNet baseline)
        expected_lr = 0.001 * (bs / 64)

        if lr > expected_lr * 3:
            return {
                'issue': True,
                'message': f'LR({lr})相对Batch Size({bs})偏大 — 可能导致震荡',
                'bs_action': 'increase',
                'suggested_bs': min(bs * 2, 512),
                'reason': f'根据线性缩放规则: batch={bs}对应最佳LR≈{expected_lr:.6f}, '
                          f'当前LR={lr}偏大。建议增大batch或降低LR。',
            }
        elif lr < expected_lr / 3:
            return {
                'issue': True,
                'message': f'LR({lr})相对Batch Size({bs})偏小 — 收敛可能过慢',
                'bs_action': 'decrease',
                'suggested_bs': max(bs // 2, 8),
                'reason': f'根据线性缩放规则: batch={bs}对应最佳LR≈{expected_lr:.6f}, '
                          f'当前LR={lr}偏小。建议减小batch或提高LR。',
            }

        return {'issue': False, 'message': 'LR-Batch Size 耦合正常'}

    @staticmethod
    def _detect_task_type(final_metrics, hyperparams):
        """检测任务类型"""
        hp_task = hyperparams.get('task_type', '')
        if hp_task in ('classification', 'regression', 'clustering'):
            return hp_task
        if 'silhouette_score' in final_metrics or 'test_silhouette_score' in final_metrics:
            return 'clustering'
        if 'r2' in final_metrics or 'test_r2' in final_metrics or 'mse' in final_metrics:
            return 'regression'
        return 'classification'

    @staticmethod
    def _detect_underfitting_v2(history, final_metrics, task_type) -> bool:
        """增强欠拟合检测"""
        if task_type == 'classification':
            train_accs = [m.get('train_accuracy', m.get('accuracy', 0)) for m in history
                          if m.get('train_accuracy') is not None or m.get('accuracy') is not None]
            test_acc = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))
            avg_train = float(np.mean(train_accs)) if train_accs else test_acc
            # 训练和测试都 < 0.55 且 gap < 0.10 → 欠拟合
            gap = abs(avg_train - test_acc) if train_accs else 0
            return avg_train < 0.55 and test_acc < 0.55 and gap < 0.10 and test_acc > 0
        elif task_type == 'regression':
            train_r2s = [m.get('train_r2', m.get('r2', 0)) for m in history
                         if m.get('train_r2') is not None or m.get('r2') is not None]
            test_r2 = final_metrics.get('test_r2', final_metrics.get('r2', 0))
            avg_train = float(np.mean(train_r2s)) if train_r2s else test_r2
            gap = abs(avg_train - test_r2) if train_r2s else 0
            return avg_train < 0.25 and test_r2 < 0.25 and gap < 0.10
        return False

    # ==================================================================
    # 智能参数推荐 (训练前)
    # ==================================================================

    _ALGO_PARAM_PRESETS = {
        'random_forest': {
            'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 'sqrt',
            '_reason': '随机森林: 200棵树平衡速度与精度, max_depth不限制让树充分生长',
        },
        'gradient_boosting': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
            'min_samples_split': 2, 'subsample': 0.8,
            '_reason': '梯度提升: 学习率0.1+深度5是大多数表格数据的甜蜜点',
        },
        'logistic_regression': {
            'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000,
            '_reason': '逻辑回归: L2正则+C=1.0, lbfgs求解器收敛快',
        },
        'svm': {
            'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale',
            '_reason': 'SVM: RBF核+自动gamma缩放, C=1.0为默认平衡值',
        },
        'knn': {
            'n_neighbors': 5, 'weights': 'distance', 'metric': 'minkowski',
            '_reason': 'KNN: K=5(奇数防平票)+距离加权',
        },
        'linear_regression': {
            'fit_intercept': True, 'positive': False,
            '_reason': '线性回归: 拟合截距, 不强制正系数保持灵活性',
        },
        'random_forest_regressor': {
            'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2,
            'min_samples_leaf': 1,
            '_reason': 'RF回归: 200棵树, 不限制深度以捕获复杂关系',
        },
        'gradient_boosting_regressor': {
            'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5,
            'subsample': 0.8,
            '_reason': 'GB回归: 学习率0.1+深度5, 稳健起点',
        },
        'mlp': {
            'hidden_layers': [128, 64, 32], 'learning_rate': 0.001,
            'dropout': 0.3, 'batch_size': 64, 'weight_decay': 1e-4,
            'early_stopping_patience': 10, 'val_size': 0.15,
            '_reason': 'MLP: 3层[128,64,32], lr=0.001, dropout=0.3 — 通用深度学习最佳实践',
        },
        'transformer_bert': {
            'learning_rate': 2e-5, 'batch_size': 16, 'max_length': 256,
            'epochs': 3, 'warmup_steps': 500, 'dropout': 0.1,
            '_reason': 'BERT微调: lr=2e-5, batch=16, 3轮epoch (Google官方推荐)',
        },
    }

    _SCALE_ADJUSTMENTS = {
        'small': {
            'n_estimators': 100, 'batch_size': 16, 'epochs': 30,
            'dropout': 0.5, 'learning_rate': 0.0005, 'weight_decay': 1e-3,
            'hidden_layers': [64, 32], 'val_size': 0.2,
        },
        'medium': {
            'n_estimators': 200, 'batch_size': 32, 'epochs': 20,
            'dropout': 0.3, 'learning_rate': 0.001, 'weight_decay': 1e-4,
            'hidden_layers': [128, 64, 32], 'val_size': 0.15,
        },
        'large': {
            'n_estimators': 200, 'batch_size': 64, 'epochs': 10,
            'dropout': 0.2, 'learning_rate': 0.001, 'weight_decay': 1e-5,
            'hidden_layers': [256, 128, 64], 'val_size': 0.1,
        },
        'xlarge': {
            'n_estimators': 100, 'batch_size': 128, 'epochs': 5,
            'dropout': 0.2, 'learning_rate': 0.001, 'weight_decay': 1e-5,
            'hidden_layers': [512, 256, 128], 'val_size': 0.05,
        },
    }

    @staticmethod
    def recommend_initial_params(analysis: dict, algorithm: str,
                                  ml_task_type: str = 'classification',
                                  framework: str = 'sklearn') -> dict:
        """训练前智能参数推荐 (保持向后兼容)"""
        n_samples = analysis.get('n_samples', 1000)
        n_features = analysis.get('n_features', 10)
        imbalanced = analysis.get('imbalanced', False)
        high_dim = analysis.get('high_dim', False)
        missing_rate = analysis.get('missing_rate', 0)

        if n_samples < 1000:
            scale = 'small'
        elif n_samples < 10000:
            scale = 'medium'
        elif n_samples < 100000:
            scale = 'large'
        else:
            scale = 'xlarge'

        scale_adj = ParameterGuidanceService._SCALE_ADJUSTMENTS[scale]
        algo_preset = ParameterGuidanceService._ALGO_PARAM_PRESETS.get(algorithm, {})
        reason = algo_preset.pop('_reason', f'{algorithm} 的通用最佳实践参数')

        params = {}
        for k, v in scale_adj.items():
            params[k] = v
        params.update(algo_preset)

        if 'hidden_layers' in params and framework == 'pytorch':
            hl = list(params['hidden_layers'])
            if n_features > 200:
                params['hidden_layers'] = [max(256, hl[0]), max(128, hl[1] if len(hl) > 1 else 64),
                                            max(64, hl[2] if len(hl) > 2 else 32)]
            elif n_features < 10:
                params['hidden_layers'] = [max(32, hl[0] // 2),
                                            max(16, hl[1] // 2 if len(hl) > 1 else 16)]

        tips = []
        if imbalanced and ml_task_type == 'classification':
            params['class_weight'] = 'balanced'
            tips.append('数据集类别不平衡, 已启用 class_weight="balanced"')
            if algorithm == 'random_forest':
                params['n_estimators'] = params.get('n_estimators', 200) + 100
                tips.append('增加树数量提升少数类识别能力')

        if high_dim:
            tips.append(f'高维数据({n_features}特征), 建议先做特征选择或PCA降维')

        if missing_rate > 0.05:
            tips.append(f'⚠ 缺失率{missing_rate:.1%}, 建议训练前做缺失值处理')

        sklearn_algos = {'random_forest', 'gradient_boosting', 'logistic_regression',
                         'svm', 'knn', 'linear_regression', 'random_forest_regressor',
                         'gradient_boosting_regressor', 'ridge', 'svr'}
        gs_suggest = algorithm in sklearn_algos and 500 < n_samples < 50000

        if gs_suggest:
            gridsearch_reason = (
                f'推荐使用 Optuna TPE Bayesian Optimization 进行自动调优。'
                f'当前数据量({n_samples:,}样本)适合在搜索空间中寻找最佳参数组合。'
                f'建议 50-100 trials, 预计耗时{_estimate_tuning_time(n_samples, len(params), algorithm)}。'
            )
        else:
            gridsearch_reason = (
                '当前推荐参数已针对数据特征优化。'
                if n_samples < 500 else
                '数据量较小, 手动按Phase 1-5顺序微调比自动搜索更高效。'
            )

        confidence = 0.90 if algorithm in sklearn_algos else 0.80

        return {
            'params': params,
            'reason': reason,
            'scale': scale,
            'confidence': confidence,
            'tips': tips,
            'gridsearch_suggestion': gs_suggest,
            'gridsearch_reason': gridsearch_reason,
        }


def _estimate_tuning_time(n_samples: int, n_params: int, algorithm: str) -> str:
    """估算Optuna调优耗时"""
    base_seconds = n_samples / 1000 * n_params * 0.3
    if algorithm in ('svm', 'svr'):
        base_seconds *= 3
    elif algorithm in ('gradient_boosting', 'gradient_boosting_regressor'):
        base_seconds *= 1.5

    if base_seconds < 30:
        return '< 30秒'
    elif base_seconds < 120:
        return '1-2分钟'
    elif base_seconds < 600:
        return f'{int(base_seconds / 60)}-{int(base_seconds / 60) + 1}分钟'
    else:
        return f'{int(base_seconds / 60)}分钟以上'
