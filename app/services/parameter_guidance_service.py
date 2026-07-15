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

import numpy as np


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
        {
            'phase': 1,
            'param': 'learning_rate',
            'name': '学习率',
            'method': 'LR Finder (指数增长扫描)',
            'reason': '单一最重要参数，决定收敛速度和最终性能',
        },
        {
            'phase': 2,
            'param': 'batch_size',
            'name': '批大小',
            'method': '线性缩放规则 (LR同步调整)',
            'reason': '与LR耦合 — 翻倍batch需翻倍LR',
        },
        {
            'phase': 3,
            'param': 'n_estimators',
            'name': '树数量/层数',
            'method': '逐步增加至收益递减',
            'reason': '控制模型容量，防止过拟合/欠拟合',
        },
        {
            'phase': 4,
            'param': 'weight_decay',
            'name': '权重衰减',
            'method': '对数尺度搜索 (1e-6 → 1e-2)',
            'reason': '精细正则化，防止过拟合',
        },
        {
            'phase': 5,
            'param': 'dropout',
            'name': 'Dropout',
            'method': '线性扫描 (0.1 → 0.7)',
            'reason': '最后一层防线 — 在LR和架构确定后微调',
        },
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
    def analyze_results(
        metrics_history: list, final_metrics: dict, dataset_info: dict = None, hyperparams: dict = None
    ) -> dict:
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
        sub_scores['absolute_perf'] = ParameterGuidanceService._score_absolute_performance_v2(final_metrics, task_type)

        # 2b. 过拟合风险 (0-100, 100=无过拟合)
        sub_scores['overfitting_risk'] = ParameterGuidanceService._score_overfitting_v2(
            metrics_history, final_metrics, task_type, hyperparams
        )

        # 2c. 收敛质量 (0-100)
        sub_scores['convergence_quality'] = ParameterGuidanceService._score_convergence_v2(
            metrics_history, task_type, final_metrics
        )

        # 2d. 类别均衡 (0-100, 100=完美均衡)
        sub_scores['class_balance'] = ParameterGuidanceService._score_class_balance_v2(final_metrics, task_type)

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
        health_score = round(sum(sub_scores[k] * weights[k] for k in weights), 1)

        # 边界校准 (防止过高/过低)
        health_score = max(3, min(99, health_score))

        # ═══════════════════════════════════════════════════════════════
        # Step 3.5: 子评分→问题映射 (补充决策树未覆盖的情况)
        # ═══════════════════════════════════════════════════════════════
        issues_from_scores = ParameterGuidanceService._issues_from_sub_scores(sub_scores, task_type, hyperparams)
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
        diagnosis = ParameterGuidanceService._generate_diagnosis_text(health_score, sub_scores, issues, task_type)

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
        train_losses = [
            m.get('train_loss', m.get('loss'))
            for m in history
            if m.get('train_loss') is not None or m.get('loss') is not None
        ]

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
                path.append({'node': 'Q1', 'question': '训练Loss是否异常高?', 'answer': 'YES', 'severity': 'high'})
                lr = hyperparams.get('learning_rate', 0.001)
                if lr > 0.01:
                    issues.append(
                        {
                            'type': 'high_initial_lr',
                            'severity': 'high',
                            'message': f'学习率({lr})可能过大导致Loss高居不下',
                            'category': 'nuisance',
                        }
                    )
                    suggestions.append(
                        {
                            'priority': 1,
                            'phase': 1,
                            'param': 'learning_rate',
                            'action': 'decrease',
                            'current': lr,
                            'suggested': round(lr / 5, 6),
                            'category': 'nuisance',
                            'reason': '学习率过高导致无法收敛 — 降低5x重试',
                            'method': 'LR Finder (指数增长扫描确定最佳LR范围)',
                        }
                    )
                else:
                    issues.append(
                        {
                            'type': 'high_loss_other',
                            'severity': 'high',
                            'message': '训练Loss异常高 — 可能数据质量问题或标签错误',
                        }
                    )
                    suggestions.append(
                        {
                            'priority': 1,
                            'phase': 0,
                            'param': 'data_quality',
                            'action': 'check',
                            'category': 'fixed',
                            'reason': '在调参之前，先排除数据质量问题: 检查缺失值/异常值/标签正确性',
                        }
                    )
            else:
                path.append(
                    {
                        'node': 'Q1',
                        'question': '训练Loss是否异常高?',
                        'answer': 'NO — Loss正常范围内',
                        'severity': 'none',
                    }
                )

        # ── Q2: Val Loss >> Train Loss? (过拟合检测) ──
        overfit_info = ParameterGuidanceService._detect_overfitting_v2(history, final_metrics, task_type, hyperparams)
        if overfit_info['overfitting']:
            sev_label = '严重' if overfit_info['severity'] == 'high' else '轻微'
            path.append(
                {
                    'node': 'Q2',
                    'question': 'Val Loss >> Train Loss?',
                    'answer': f'YES — {sev_label}过拟合 (gap={overfit_info.get("gap", "?")})',
                    'severity': overfit_info['severity'],
                }
            )

            # 分级别建议
            if overfit_info['severity'] == 'high':
                suggestions.append(
                    {
                        'priority': 1,
                        'phase': 5,
                        'param': 'dropout',
                        'action': 'increase',
                        'current': hyperparams.get('dropout', 0.3),
                        'suggested': min(0.7, hyperparams.get('dropout', 0.3) + 0.3),
                        'category': 'nuisance',
                        'reason': '严重过拟合 — 大幅增加Dropout是最直接有效的正则化手段',
                    }
                )
                suggestions.append(
                    {
                        'priority': 2,
                        'phase': 4,
                        'param': 'weight_decay',
                        'action': 'increase',
                        'current': hyperparams.get('weight_decay', 1e-4),
                        'suggested': hyperparams.get('weight_decay', 1e-4) * 10,
                        'category': 'nuisance',
                        'reason': '配合增大weight_decay形成双重正则化防线',
                    }
                )
            else:
                suggestions.append(
                    {
                        'priority': 3,
                        'phase': 5,
                        'param': 'dropout',
                        'action': 'increase',
                        'current': hyperparams.get('dropout', 0.3),
                        'suggested': min(0.5, hyperparams.get('dropout', 0.3) + 0.15),
                        'category': 'nuisance',
                        'reason': '轻微过拟合 — 适度增加Dropout即可',
                    }
                )
        else:
            path.append(
                {
                    'node': 'Q2',
                    'question': 'Val Loss >> Train Loss?',
                    'answer': 'NO — Train/Val 差距正常',
                    'severity': 'none',
                }
            )

        # ── Q3: 欠拟合检测 ──
        is_underfit = ParameterGuidanceService._detect_underfitting_v2(history, final_metrics, task_type)
        if is_underfit:
            path.append(
                {'node': 'Q3', 'question': 'Train ≈ Val 但都很低?', 'answer': 'YES — 欠拟合', 'severity': 'medium'}
            )
            suggestions.append(
                {
                    'priority': 2,
                    'phase': 3,
                    'param': 'hidden_layers',
                    'action': 'increase',
                    'current': hyperparams.get('hidden_layers', [128, 64]),
                    'suggested': [256, 128, 64, 32],
                    'category': 'scientific',
                    'reason': '模型容量不足 — 增加网络深度/宽度提升表达能力',
                }
            )
            suggestions.append(
                {
                    'priority': 1,
                    'phase': 3,
                    'param': 'algorithm',
                    'action': 'change',
                    'suggested': 'gradient_boosting',
                    'category': 'scientific',
                    'reason': '当前算法容量不足，建议切换到更强的集成方法(GB/RF)',
                }
            )
        else:
            path.append({'node': 'Q3', 'question': 'Train ≈ Val 但都很低?', 'answer': 'NO', 'severity': 'none'})

        # ── Q4: Loss震荡 / 收敛速度? ──
        loss_analysis = ParameterGuidanceService._analyze_loss_curve_v2(history)
        if loss_analysis['trend'] == 'unknown' and len(history) < 3:
            # 单轮模型: 无足够epoch数据做loss曲线分析
            path.append(
                {
                    'node': 'Q4',
                    'question': 'Loss收敛情况?',
                    'answer': 'ⓘ 单轮训练, 无Loss曲线数据 — 跳过收敛分析',
                    'severity': 'none',
                }
            )
        elif loss_analysis['oscillating']:
            path.append(
                {
                    'node': 'Q4',
                    'question': 'Loss是否剧烈震荡?',
                    'answer': f'YES — CV={loss_analysis["cv"]:.2f}',
                    'severity': 'medium',
                }
            )
            current_lr = hyperparams.get('learning_rate', 0.001)
            current_bs = hyperparams.get('batch_size', 64)
            suggested_lr = round(max(current_lr / 5, 1e-6), 6)
            suggestions.append(
                {
                    'priority': 1,
                    'phase': 1,
                    'param': 'learning_rate',
                    'action': 'decrease',
                    'current': current_lr,
                    'suggested': suggested_lr,
                    'category': 'nuisance',
                    'reason': f'Loss震荡(CV={loss_analysis["cv"]:.1%}) → 学习率过大。'
                    f'标准修复: LR从{current_lr}降至{suggested_lr}',
                }
            )
            if current_bs < 64:
                suggestions.append(
                    {
                        'priority': 3,
                        'phase': 2,
                        'param': 'batch_size',
                        'action': 'increase',
                        'current': current_bs,
                        'suggested': min(current_bs * 2, 256),
                        'category': 'nuisance',
                        'reason': '增大batch size可平滑梯度估计，减少震荡 (LR需同步缩放)',
                    }
                )
        elif loss_analysis['slow_convergence']:
            path.append({'node': 'Q4', 'question': 'Loss是否剧烈震荡?', 'answer': 'NO — 但收敛偏慢', 'severity': 'low'})
            current_lr = hyperparams.get('learning_rate', 0.001)
            suggested_lr = round(current_lr * 3, 6)
            suggestions.append(
                {
                    'priority': 4,
                    'phase': 1,
                    'param': 'learning_rate',
                    'action': 'increase',
                    'current': current_lr,
                    'suggested': suggested_lr,
                    'category': 'nuisance',
                    'reason': '收敛偏慢 — 可适当提高LR加速训练。运行LR Finder确定最佳范围',
                }
            )
        else:
            path.append({'node': 'Q4', 'question': 'Loss是否剧烈震荡?', 'answer': 'NO — 收敛稳定', 'severity': 'none'})

        # ── Q5: LR-Batch Size 耦合检查 ──
        if has_enough_history:
            lr_bs_info = ParameterGuidanceService._check_lr_batch_coupling(history, hyperparams)
            if lr_bs_info['issue']:
                path.append(
                    {
                        'node': 'Q5',
                        'question': 'LR-Batch Size 耦合是否合理?',
                        'answer': f'问题: {lr_bs_info["message"]}',
                        'severity': 'medium',
                    }
                )
                suggestions.append(
                    {
                        'priority': 5,
                        'phase': 2,
                        'param': 'batch_size',
                        'action': lr_bs_info['bs_action'],
                        'suggested': lr_bs_info['suggested_bs'],
                        'category': 'nuisance',
                        'reason': lr_bs_info['reason'],
                    }
                )

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
            if acc >= 0.98:
                base = 78 + (acc - 0.98) / 0.02 * 2
            elif acc >= 0.95:
                base = 68 + (acc - 0.95) / 0.03 * 10
            elif acc >= 0.88:
                base = 55 + (acc - 0.88) / 0.07 * 13
            elif acc >= 0.80:
                base = 40 + (acc - 0.80) / 0.08 * 15
            elif acc >= 0.70:
                base = 25 + (acc - 0.70) / 0.10 * 15
            elif acc >= 0.55:
                base = 10 + (acc - 0.55) / 0.15 * 15
            elif acc >= 0.40:
                base = 3 + (acc - 0.40) / 0.15 * 7
            elif acc > 0:
                base = max(0, acc * 7)
            else:
                base = 0

            # F1 → ±15 调整
            if f1 > 0:
                if f1 >= 0.95:
                    bonus = 15
                elif f1 >= 0.85:
                    bonus = 10
                elif f1 >= 0.75:
                    bonus = 5
                elif f1 >= 0.60:
                    bonus = 0
                elif f1 >= 0.45:
                    bonus = -5
                elif f1 >= 0.30:
                    bonus = -10
                else:
                    bonus = -15
            else:
                bonus = 0

            return round(max(0, min(100, base + bonus)), 1)

        elif task_type == 'regression':
            r2 = final_metrics.get('test_r2', final_metrics.get('r2', None))

            if r2 is not None:
                if r2 >= 0.95:
                    score = 92 + (r2 - 0.95) / 0.05 * 8
                elif r2 >= 0.85:
                    score = 78 + (r2 - 0.85) / 0.10 * 14
                elif r2 >= 0.70:
                    score = 60 + (r2 - 0.70) / 0.15 * 18
                elif r2 >= 0.50:
                    score = 38 + (r2 - 0.50) / 0.20 * 22
                elif r2 >= 0.25:
                    score = 18 + (r2 - 0.25) / 0.25 * 20
                elif r2 >= 0:
                    score = 5 + r2 / 0.25 * 13
                else:
                    score = max(0, 3 + r2 * 3)
                return round(max(0, min(100, score)), 1)
            return 20.0  # 无R²指标

        elif task_type == 'clustering':
            sil = final_metrics.get('test_silhouette_score', final_metrics.get('silhouette_score', None))
            if sil is not None:
                if sil >= 0.70:
                    score = 88 + (sil - 0.70) / 0.30 * 12
                elif sil >= 0.50:
                    score = 65 + (sil - 0.50) / 0.20 * 23
                elif sil >= 0.30:
                    score = 38 + (sil - 0.30) / 0.20 * 27
                elif sil >= 0.10:
                    score = 15 + (sil - 0.10) / 0.20 * 23
                elif sil >= 0:
                    score = 5 + sil / 0.10 * 10
                else:
                    score = max(0, 2 + sil * 3)
                return round(max(0, min(100, score)), 1)
            return 20.0

        return 20.0

    # ==================================================================
    # Step 2b: 过拟合风险评分 v2
    # ==================================================================

    @staticmethod
    def _score_overfitting_v2(history, final_metrics, task_type, hyperparams: dict = None) -> float:
        """过拟合风险评分 (100 = 完全无过拟合, 0 = 严重过拟合)"""
        info = ParameterGuidanceService._detect_overfitting_v2(history, final_metrics, task_type, hyperparams)
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
    def _detect_overfitting_v2(history, final_metrics, task_type, hyperparams: dict = None) -> dict:
        """增强过拟合检测 — 多信号融合

        对单轮 sklearn 模型(无独立 train/test 指标):
          使用 CV 折间方差 + best_cv_score vs test_score 对比作为替代信号
        """
        hyperparams = hyperparams or {}
        if len(history) < 1:
            return {'overfitting': False, 'severity': 'none', 'gap': 0}

        n_recent = max(1, len(history) // 3)
        recent = history[-n_recent:]

        if task_type == 'regression':
            train_key = 'train_r2'
            train_vals = [m.get(train_key) for m in recent if m.get(train_key) is not None]
            test_val = final_metrics.get('test_r2', final_metrics.get('r2'))
            # 无独立 train 指标时回退到通用指标
            if not train_vals:
                train_vals = [m.get('r2') for m in recent if m.get('r2') is not None]
        elif task_type == 'clustering':
            return {'overfitting': False, 'severity': 'none', 'gap': 0}
        else:
            train_key = 'train_accuracy'
            train_vals = [m.get(train_key) for m in recent if m.get(train_key) is not None]
            test_val = final_metrics.get('test_accuracy', final_metrics.get('accuracy'))
            # 无独立 train 指标时回退到通用指标
            if not train_vals:
                train_vals = [m.get('accuracy') for m in recent if m.get('accuracy') is not None]

        # ── 检测是否有真正分离的 train/test 指标 ──
        has_separated = any(m.get(train_key) is not None for m in recent if train_key in ('train_accuracy', 'train_r2'))

        if not train_vals or test_val is None:
            # 无任何指标 → 尝试 CV 方差信号
            return ParameterGuidanceService._overfitting_from_cv_signal(hyperparams, final_metrics, task_type)

        avg_train = float(np.mean(train_vals))
        gap = avg_train - test_val

        # ── 当 train/test 来源相同时, gap 天然≈0 → 需用 CV 信号补充 ──
        if not has_separated and abs(gap) < 0.005:
            cv_result = ParameterGuidanceService._overfitting_from_cv_signal(hyperparams, final_metrics, task_type)
            if cv_result['overfitting']:
                return cv_result
            # 无过拟合信号 → 返回标注 inconclusive 的结果
            return {
                'overfitting': False,
                'severity': 'none',
                'gap': 0,
                'note': 'inconclusive — 单轮模型无独立 train/test 分离指标, 过拟合检测基于 CV 方差推断',
            }

        if task_type == 'regression':
            if gap > 0.30:
                return {'overfitting': True, 'severity': 'critical', 'gap': round(gap, 4)}
            elif gap > 0.20:
                return {'overfitting': True, 'severity': 'high', 'gap': round(gap, 4)}
            elif gap > 0.10:
                return {'overfitting': True, 'severity': 'medium', 'gap': round(gap, 4)}
            elif gap > 0.05:
                return {'overfitting': True, 'severity': 'low', 'gap': round(gap, 4)}
        else:
            if gap > 0.20:
                return {'overfitting': True, 'severity': 'critical', 'gap': round(gap, 4)}
            elif gap > 0.12:
                return {'overfitting': True, 'severity': 'high', 'gap': round(gap, 4)}
            elif gap > 0.06:
                return {'overfitting': True, 'severity': 'medium', 'gap': round(gap, 4)}
            elif gap > 0.03:
                return {'overfitting': True, 'severity': 'low', 'gap': round(gap, 4)}

        return {'overfitting': False, 'severity': 'none', 'gap': round(gap, 4)}

    @staticmethod
    def _overfitting_from_cv_signal(hyperparams: dict, final_metrics: dict, task_type: str) -> dict:
        """从 CV 调优结果推断过拟合信号 (替代 train/test gap)"""
        tuning = hyperparams.get('tuning_result', {})
        best_cv = tuning.get('best_cv_score') if isinstance(tuning, dict) else None
        if best_cv is None:
            best_cv = hyperparams.get('best_cv_score')

        if best_cv is None:
            return {'overfitting': False, 'severity': 'none', 'gap': 0}

        # 获取最终测试集分数
        if task_type == 'regression':
            test_score = final_metrics.get('test_r2', final_metrics.get('r2', 0))
        elif task_type == 'clustering':
            test_score = final_metrics.get('test_silhouette_score', final_metrics.get('silhouette_score', 0))
        else:
            test_score = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))

        gap = best_cv - test_score

        # CV 分数是交叉验证均值, test 分数是留出集单次评估
        # CV > test 说明模型在训练数据的不同子集上表现不一致 → 过拟合信号
        if task_type == 'regression':
            if gap > 0.25:
                return {'overfitting': True, 'severity': 'critical', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
            elif gap > 0.15:
                return {'overfitting': True, 'severity': 'high', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
            elif gap > 0.08:
                return {'overfitting': True, 'severity': 'medium', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
            elif gap > 0.04:
                return {'overfitting': True, 'severity': 'low', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
        else:
            if gap > 0.15:
                return {'overfitting': True, 'severity': 'critical', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
            elif gap > 0.10:
                return {'overfitting': True, 'severity': 'high', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
            elif gap > 0.05:
                return {'overfitting': True, 'severity': 'medium', 'gap': round(gap, 4), 'source': 'cv_vs_test'}
            elif gap > 0.02:
                return {'overfitting': True, 'severity': 'low', 'gap': round(gap, 4), 'source': 'cv_vs_test'}

        return {'overfitting': False, 'severity': 'none', 'gap': round(gap, 4)}

    # ==================================================================
    # Step 2c: 收敛质量评分 v2
    # ==================================================================

    @staticmethod
    def _score_convergence_v2(history, task_type, final_metrics: dict = None) -> float:
        """收敛质量评分 (100 = 完美收敛, 0 = 完全未收敛)

        对于单轮模型(sklearn 1-epoch): 无法分析Loss曲线, 基于最终指标推算收敛质量
        """
        n = len(history)
        final_metrics = final_metrics or {}

        if n < 3:
            # 单轮模型: 从最终指标反推收敛质量
            entry = history[-1] if n >= 1 else final_metrics

            if task_type == 'classification':
                test_acc = entry.get(
                    'test_accuracy',
                    entry.get('accuracy', final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))),
                )
                entry.get('train_accuracy', test_acc)
                if test_acc >= 0.90:
                    return 85.0  # 高准确率 → 收敛良好
                elif test_acc >= 0.78:
                    return 73.0  # 中等 → 可接受
                elif test_acc >= 0.60:
                    return 58.0  # 偏低 → 可能欠收敛
                elif test_acc > 0:
                    return 40.0  # 低 → 收敛差
                else:
                    return 30.0  # 无有效指标
            elif task_type == 'regression':
                r2 = entry.get('test_r2', entry.get('r2', final_metrics.get('test_r2', final_metrics.get('r2'))))
                if r2 is not None:
                    if r2 >= 0.85:
                        return 85.0
                    elif r2 >= 0.65:
                        return 73.0
                    elif r2 >= 0.40:
                        return 58.0
                    elif r2 >= 0.10:
                        return 40.0
                    else:
                        return 30.0
                return 50.0  # 无 R² 指标
            elif task_type == 'clustering':
                sil = entry.get(
                    'test_silhouette_score',
                    entry.get(
                        'silhouette_score',
                        final_metrics.get('test_silhouette_score', final_metrics.get('silhouette_score')),
                    ),
                )
                if sil is not None:
                    if sil >= 0.60:
                        return 85.0
                    elif sil >= 0.40:
                        return 73.0
                    elif sil >= 0.20:
                        return 58.0
                    elif sil >= 0.05:
                        return 40.0
                    else:
                        return 30.0
                return 50.0
            return 50.0  # 完全无法评估

        analysis = ParameterGuidanceService._analyze_loss_curve_v2(history)

        # 基础分 + 扣分项
        score = 80.0

        if analysis['oscillating']:
            cv = analysis['cv']
            if cv > 0.50:
                score -= 40
            elif cv > 0.35:
                score -= 30
            elif cv > 0.25:
                score -= 20
            else:
                score -= 10

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
            return {
                'oscillating': False,
                'slow_convergence': False,
                'converged_well': False,
                'diverging': False,
                'cv': 0,
                'trend': 'unknown',
            }

        losses = [
            m.get('loss', m.get('train_loss'))
            for m in history
            if m.get('loss') is not None or m.get('train_loss') is not None
        ]
        if len(losses) < 3:
            return {
                'oscillating': False,
                'slow_convergence': False,
                'converged_well': False,
                'diverging': False,
                'cv': 0,
                'trend': 'unknown',
            }

        avg_loss = np.mean(losses)
        if avg_loss <= 0:
            return {
                'oscillating': False,
                'slow_convergence': False,
                'converged_well': False,
                'diverging': False,
                'cv': 0,
                'trend': 'unknown',
            }

        # 变异系数 (CV = std/mean)
        cv = float(np.std(losses) / avg_loss)

        # 趋势检测 (线性回归斜率)
        x = np.arange(len(losses))
        slope = float(np.polyfit(x, losses, 1)[0])

        # 发散检测: 后期Loss > 初期Loss * 1.5
        early_mean = float(np.mean(losses[: min(3, len(losses))]))
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
            'trend': 'diverging'
            if diverging
            else 'converging'
            if slope < -0.01
            else 'flat'
            if abs(slope) < 0.01
            else 'increasing',
            'slope': round(float(slope), 6),
            'early_mean': round(early_mean, 4),
            'late_mean': round(late_mean, 4),
        }

    # ==================================================================
    # Step 2d: 类别均衡评分 v2
    # ==================================================================

    @staticmethod
    def _score_class_balance_v2(final_metrics, task_type) -> float:
        """类别均衡评分 (100 = 完美均衡)

        分类: Precision/Recall 差距 + F1-Accuracy 背离度
        回归: 残差分布偏度 (symmetry of errors)
        聚类: 簇大小均衡度 (max/min cluster ratio)
        """
        if task_type == 'regression':
            return ParameterGuidanceService._score_residual_balance_v2(final_metrics)
        if task_type == 'clustering':
            return ParameterGuidanceService._score_cluster_balance_v2(final_metrics)

        prec = final_metrics.get('test_precision', final_metrics.get('precision', 0))
        rec = final_metrics.get('test_recall', final_metrics.get('recall', 0))
        f1 = final_metrics.get('test_f1_score', final_metrics.get('f1_score', 0))
        acc = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))

        score = 100.0

        # P-R 差距过大
        if prec > 0 and rec > 0:
            pr_gap = abs(prec - rec)
            if pr_gap > 0.40:
                score -= 35
            elif pr_gap > 0.25:
                score -= 20
            elif pr_gap > 0.15:
                score -= 10
            elif pr_gap > 0.08:
                score -= 5

        # F1 远低于 Accuracy (类别不平衡的标志)
        if acc > 0 and f1 > 0:
            acc_f1_gap = acc - f1
            if acc_f1_gap > 0.30:
                score -= 25
            elif acc_f1_gap > 0.15:
                score -= 12
            elif acc_f1_gap > 0.08:
                score -= 5

        # 极端P或R
        if prec > 0 and prec < 0.30:
            score -= 15
        if rec > 0 and rec < 0.30:
            score -= 15

        return round(max(0, min(100, score)), 1)

    @staticmethod
    def _score_residual_balance_v2(final_metrics) -> float:
        """回归残差均衡评分 — 基于残差分布偏度

        信号:
          - residual_skew: 残差偏度 (0=完美对称, |skew|>1=严重偏斜)
          - residual_kurtosis: 残差峰度 (>3=厚尾, 模型对大误差样本预测差)
          - RMSE/MAE 比值: 远大于1说明存在异常大误差样本
        """
        skew = abs(final_metrics.get('residual_skew', final_metrics.get('residual_skewness', 0)))
        kurt = final_metrics.get('residual_kurtosis', 0)
        rmse = final_metrics.get('test_rmse', final_metrics.get('rmse', 0))
        mae = final_metrics.get('test_mae', final_metrics.get('mae', 1))

        score = 85.0  # 基础分 (比分类保守 — 回归的均衡概念不同)

        # 残差偏度惩罚
        if skew > 1.5:
            score -= 35
        elif skew > 1.0:
            score -= 20
        elif skew > 0.5:
            score -= 10
        elif skew > 0.25:
            score -= 5

        # 厚尾惩罚 (模型在某些样本上表现显著更差)
        if kurt > 4.0:
            score -= 20
        elif kurt > 3.5:
            score -= 10

        # RMSE/MAE 比值 — 远大于1说明有异常大误差样本
        if mae > 0 and rmse > 0:
            ratio = rmse / mae
            if ratio > 2.5:
                score -= 15
            elif ratio > 1.8:
                score -= 8

        return round(max(0, min(100, score)), 1)

    @staticmethod
    def _score_cluster_balance_v2(final_metrics) -> float:
        """聚类簇均衡评分 — 基于簇大小分布

        信号:
          - cluster_sizes: 各簇样本数列表
          - cluster_balance_ratio: min/max 簇大小比 (>0.5=均衡, <0.1=严重不均衡)
          - noise_ratio: DBSCAN 噪声点比例
        """
        cluster_sizes = final_metrics.get('cluster_sizes', None)
        noise_ratio = final_metrics.get('noise_ratio', final_metrics.get('noise_points_ratio', 0))

        if cluster_sizes and len(cluster_sizes) >= 2:
            sizes = [s for s in cluster_sizes if s > 0]
            if len(sizes) >= 2:
                balance_ratio = min(sizes) / max(sizes) if max(sizes) > 0 else 0
                if balance_ratio >= 0.7:
                    score = 95.0
                elif balance_ratio >= 0.5:
                    score = 85.0
                elif balance_ratio >= 0.3:
                    score = 65.0
                elif balance_ratio >= 0.15:
                    score = 40.0
                elif balance_ratio >= 0.05:
                    score = 20.0
                else:
                    score = 8.0
            else:
                score = 30.0  # 仅1个有效簇 — 几乎肯定有问题
        else:
            # 无簇大小数据 → 从 silhouette 推算
            sil = final_metrics.get('test_silhouette_score', final_metrics.get('silhouette_score', 0.3))
            n_clusters = final_metrics.get('n_clusters', 3)
            base = sil * 100
            if n_clusters > 10:
                base = min(100, base * 1.3)  # 多簇场景容错
            score = max(10, min(95, base))

        # 噪声点惩罚 (DBSCAN)
        if noise_ratio > 0.30:
            score -= 30
        elif noise_ratio > 0.15:
            score -= 15
        elif noise_ratio > 0.08:
            score -= 8

        return round(max(0, min(100, score)), 1)

    # ==================================================================
    # Step 2e: 泛化能力评分 v2
    # ==================================================================

    @staticmethod
    def _score_generalization_v2(history, final_metrics, task_type, hyperparams, dataset_info) -> float:
        """泛化能力评分 (100 = 完美泛化)"""
        score = 80.0  # 基础分

        # Train-Test gap 信号
        n = len(history)
        if n >= 2:
            if task_type == 'classification':
                train_vals = [
                    m.get('train_accuracy', m.get('accuracy'))
                    for m in history
                    if m.get('train_accuracy') is not None or m.get('accuracy') is not None
                ]
                test_val = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))
                if train_vals:
                    avg_train = float(np.mean(train_vals))
                    gap = avg_train - test_val
                    if gap > 0.15:
                        score -= 30
                    elif gap > 0.08:
                        score -= 15
                    elif gap > 0.03:
                        score -= 5
            elif task_type == 'regression':
                train_r2s = [
                    m.get('train_r2', m.get('r2'))
                    for m in history
                    if m.get('train_r2') is not None or m.get('r2') is not None
                ]
                test_r2 = final_metrics.get('test_r2', final_metrics.get('r2', 0))
                if train_r2s:
                    avg_train = float(np.mean(train_r2s))
                    gap = avg_train - test_r2
                    if gap > 0.20:
                        score -= 30
                    elif gap > 0.10:
                        score -= 15
                    elif gap > 0.05:
                        score -= 5

        # 模型容量-数据量匹配 (如果可用)
        n_samples = dataset_info.get('n_samples', 0) if dataset_info else 0
        hl = hyperparams.get('hidden_layers', [])
        if hl and n_samples > 0:
            total_params_est = sum(hl) * 100  # 粗略估计
            ratio = n_samples / max(total_params_est, 1)
            if ratio < 1:
                score -= 15  # 参数多于样本 → 高风险过拟合
            elif ratio < 5:
                score -= 5
            elif ratio > 50:
                score += 5  # 充足数据

        return round(max(0, min(100, score)), 1)

    # ==================================================================
    # Step 3.5: 子评分→问题映射
    # ==================================================================

    @staticmethod
    def _issues_from_sub_scores(sub_scores, task_type, hyperparams) -> list:
        """从子评分反推问题 — 梯度阈值映射 (v2.2)

        使用三层阈值替代单一切断点:
          - score < 45  → high severity   (严重问题, 必须处理)
          - score < 65  → medium severity (需要关注, 建议优化)
          - score < 80  → low severity    (可改进, 非阻塞)
          - score >= 80 → 不生成 issue    (该维度健康)
        """
        issues = []
        dim_meta = {
            'absolute_perf': ('绝对性能不足', '模型预测能力未达到可用水平'),
            'overfitting_risk': ('存在过拟合风险', '训练/验证集指标差距偏大'),
            'convergence_quality': ('收敛质量偏低', 'Loss曲线收敛不理想或无法评估(单轮模型)'),
            'class_balance': ('类别不均衡影响', 'Precision/Recall差距偏大'),
            'generalization': ('泛化能力不足', '模型在新数据上表现可能不稳定'),
        }

        for dim, (title, desc) in dim_meta.items():
            score = sub_scores.get(dim, 100)
            if score < 45:
                severity = 'high'
                level_text = '严重'
            elif score < 65:
                severity = 'medium'
                level_text = '中等'
            elif score < 80:
                severity = 'low'
                level_text = '轻微'
            else:
                continue  # 健康, 不生成 issue

            issues.append(
                {
                    'type': f'low_{dim}',
                    'severity': severity,
                    'message': f'{title} ({level_text}): {desc} (评分 {score:.0f}/100)',
                    'category': 'scientific' if dim == 'absolute_perf' else 'nuisance',
                }
            )
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
            final_metrics.get('accuracy')
            or final_metrics.get('f1_score')
            or final_metrics.get('r2') is not None
            or final_metrics.get('silhouette_score')
            or final_metrics.get('test_accuracy')
            or final_metrics.get('test_f1_score')
            or final_metrics.get('test_r2') is not None
            or final_metrics.get('test_silhouette_score')
        )
        no_tuned = not hyperparams.get('tuned')
        no_baseline = not hyperparams.get('baseline_score')
        no_metrics_and_history = not has_meaningful_metrics and len(history) < 2
        if no_tuned and no_baseline and no_metrics_and_history:
            found.append({
                'type': 'no_baseline',
                'name': '缺少有效Baseline',
                'severity': 'medium',
                'message': '未检测到有效指标 — 模型可能未成功训练或指标为空',
                'fix': '检查训练日志，确认模型成功完成训练并产生了指标',
            })

        # 反模式2: GridSearch高维
        if hyperparams.get('tuning_method') == 'grid' and hyperparams.get('tuned'):
            hp_count = len(
                [
                    k
                    for k in hyperparams
                    if k
                    not in (
                        'task_type',
                        'algorithm',
                        'target_column',
                        'tuned',
                        'tuning_method',
                        'best_cv_score',
                        'tuning_cv_folds',
                    )
                ]
            )
            if hp_count > 3:
                found.append(
                    {
                        'type': 'grid_search_high_dim',
                        'name': '高维Grid Search',
                        'severity': 'medium',
                        'message': f'对{hp_count}个超参数使用Grid Search — 推荐切换到Optuna TPE Bayesian Optimization',
                        'fix': '使用Optuna进行贝叶斯优化，或至少切换到Random Search',
                    }
                )

        # 反模式3: 测试集泄露风险
        if final_metrics:
            metric_keys = set(final_metrics.keys())
            suspicious = {'test_accuracy', 'test_f1', 'test_r2'} & metric_keys
            if suspicious and len(history) > 10:
                # 多轮调整后还在报告test指标 → 可能存在test调参
                found.append(
                    {
                        'type': 'potential_test_leak',
                        'name': '潜在的测试集过拟合',
                        'severity': 'low',
                        'message': '警告: 经过多轮训练后仍报告test指标 — 确保test集仅用于最终评估',
                        'fix': '采用嵌套CV: 外层CV评估泛化性能，内层CV选择超参数',
                    }
                )

        return found

    # ==================================================================
    # Step 5: 系统性调参建议
    # ==================================================================

    @staticmethod
    def _generate_scientific_suggestions(issues, sub_scores, task_type, hyperparams, health_score) -> list:
        """按科学调参顺序生成建议 (Phase 1→5)

        v2.1: 降低触发阈值 + 算法感知 — 不再对树模型建议dropout/batch_size
        """
        suggestions = []
        algorithm = hyperparams.get('algorithm', '')
        algo_lower = algorithm.lower()

        # 算法分类 (决定哪些建议有意义)
        _TREE_ALGOS = {
            'random_forest',
            'random_forest_regressor',
            'gradient_boosting',
            'gradient_boosting_regressor',
            'decision_tree',
        }
        _LINEAR_ALGOS = {'logistic_regression', 'linear_regression', 'ridge', 'svm', 'svr'}
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
                suggestions.append(
                    {
                        'priority': 1,
                        'phase': 1,
                        'param': 'learning_rate',
                        'action': 'decrease',
                        'current': current_lr,
                        'suggested': round(max(current_lr / 3, 1e-6), 6),
                        'category': 'nuisance',
                        'reason': f'收敛质量评分偏低({sub_scores["convergence_quality"]:.0f}/100) — '
                        f'降低GBDT学习率+增加n_estimators补偿',
                        'method': 'GBDT: 学习率减半 → n_estimators加倍',
                    }
                )
            elif is_dl:
                suggestions.append(
                    {
                        'priority': 1,
                        'phase': 1,
                        'param': 'learning_rate',
                        'action': 'decrease',
                        'current': current_lr,
                        'suggested': round(max(current_lr / 5, 1e-6), 6),
                        'category': 'nuisance',
                        'reason': f'收敛质量评分偏低({sub_scores["convergence_quality"]:.0f}/100) — '
                        f'优先调整学习率 (Phase 1: 最重要的参数)',
                        'method': '运行LR Finder确定最佳LR范围',
                    }
                )

        # 过拟合风险高 → 正则化 (不同算法有不同正则化手段)
        if sub_scores.get('overfitting_risk', 100) < 75:
            if is_tree:
                suggestions.append(
                    {
                        'priority': 2,
                        'phase': 3,
                        'param': 'max_depth',
                        'action': 'decrease',
                        'current': hyperparams.get('max_depth', 'None'),
                        'suggested': hyperparams.get('max_depth') or 10,
                        'category': 'scientific',
                        'reason': f'过拟合风险评分({sub_scores["overfitting_risk"]:.0f}/100) — '
                        f'树模型: 限制max_depth是最有效的正则化手段',
                    }
                )
                suggestions.append(
                    {
                        'priority': 3,
                        'phase': 3,
                        'param': 'min_samples_split',
                        'action': 'increase',
                        'current': hyperparams.get('min_samples_split', 2),
                        'suggested': min(hyperparams.get('min_samples_split', 2) * 3, 20),
                        'category': 'scientific',
                        'reason': '增大min_samples_split进一步防止过拟合',
                    }
                )
            elif is_dl:
                suggestions.append(
                    {
                        'priority': 2,
                        'phase': 4,
                        'param': 'weight_decay',
                        'action': 'increase',
                        'current': hyperparams.get('weight_decay', 1e-4),
                        'suggested': hyperparams.get('weight_decay', 1e-4) * 5,
                        'category': 'nuisance',
                        'reason': f'过拟合风险评分({sub_scores["overfitting_risk"]:.0f}/100) — '
                        f'Phase 4: 增强weight_decay正则化',
                    }
                )
                suggestions.append(
                    {
                        'priority': 3,
                        'phase': 5,
                        'param': 'dropout',
                        'action': 'increase',
                        'current': hyperparams.get('dropout', 0.3),
                        'suggested': min(0.7, hyperparams.get('dropout', 0.3) + 0.2),
                        'category': 'nuisance',
                        'reason': '配合增加dropout形成双重正则化防线',
                    }
                )
            elif not is_cluster:
                # SVM/线性模型: 增强C/alpha
                suggestions.append(
                    {
                        'priority': 2,
                        'phase': 4,
                        'param': 'C' if 'svm' in algo_lower else 'alpha',
                        'action': 'decrease',
                        'current': hyperparams.get('C', hyperparams.get('alpha', 1.0)),
                        'suggested': round(hyperparams.get('C', hyperparams.get('alpha', 1.0)) / 5, 4),
                        'category': 'nuisance',
                        'reason': f'过拟合风险评分({sub_scores["overfitting_risk"]:.0f}/100) — '
                        f'增强L2正则化 (减小C/alpha = 更强正则化)',
                    }
                )

        # 类别不均衡 → class_weight
        if sub_scores.get('class_balance', 100) < 80 and task_type == 'classification':
            suggestions.append(
                {
                    'priority': 3,
                    'phase': 1,
                    'param': 'class_weight',
                    'action': 'enable',
                    'suggested': 'balanced',
                    'category': 'scientific',
                    'reason': f'类别均衡评分偏低({sub_scores["class_balance"]:.0f}/100) — '
                    f'启用class_weight="balanced"处理类别不平衡',
                }
            )

        # 绝对性能低 → 提升模型容量
        if sub_scores.get('absolute_perf', 100) < 70:
            if is_tree:
                suggestions.append(
                    {
                        'priority': 4,
                        'phase': 3,
                        'param': 'n_estimators',
                        'action': 'increase',
                        'current': hyperparams.get('n_estimators', 200),
                        'suggested': min(hyperparams.get('n_estimators', 200) * 2, 500),
                        'category': 'scientific',
                        'reason': f'绝对性能评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                        f'增加树数量提升模型表达能力',
                    }
                )
            elif is_dl:
                hl = hyperparams.get('hidden_layers', [128, 64, 32])
                wider = [min(h * 2, 1024) for h in hl]
                suggestions.append(
                    {
                        'priority': 4,
                        'phase': 3,
                        'param': 'hidden_layers',
                        'action': 'widen',
                        'current': hl,
                        'suggested': wider,
                        'category': 'scientific',
                        'reason': f'绝对性能评分偏低({sub_scores["absolute_perf"]:.0f}/100) — 加宽网络提升模型容量',
                    }
                )
            elif is_cluster:
                suggestions.append(
                    {
                        'priority': 4,
                        'phase': 3,
                        'param': 'algorithm',
                        'action': 'change',
                        'suggested': 'kmeans' if algo_lower != 'kmeans' else 'agglomerative',
                        'category': 'scientific',
                        'reason': f'聚类质量评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                        f'尝试不同聚类算法。K-Means适合球形簇，DBSCAN适合任意形状',
                    }
                )
            elif algo_lower in _LINEAR_ALGOS:
                suggestions.append(
                    {
                        'priority': 4,
                        'phase': 3,
                        'param': 'algorithm',
                        'action': 'change',
                        'suggested': 'random_forest',
                        'category': 'scientific',
                        'reason': f'绝对性能评分偏低({sub_scores["absolute_perf"]:.0f}/100) — '
                        f'线性模型容量有限，建议升级到RandomForest/GBDT',
                    }
                )

        # 泛化能力低
        if sub_scores.get('generalization', 100) < 75:
            n_samples = hyperparams.get('_n_samples', 0)
            if n_samples < 1000:
                suggestions.append(
                    {
                        'priority': 5,
                        'phase': 5,
                        'param': 'test_size',
                        'action': 'increase',
                        'current': hyperparams.get('test_size', 0.2),
                        'suggested': 0.3,
                        'category': 'fixed',
                        'reason': f'泛化能力评分偏低({sub_scores["generalization"]:.0f}/100) + '
                        f'小样本({n_samples}) — 增加测试集比例以获得更可靠的泛化估计',
                    }
                )

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
        weak_dims = [(dim_names.get(k, k), v) for k, v in sub_scores.items() if v < 80 and k != 'overfitting_risk']
        weak_dims_str = '、'.join(f'{name}({score:.0f}/100)' for name, score in weak_dims[:3]) if weak_dims else '无'

        task_cn = {'classification': '分类', 'regression': '回归', 'clustering': '聚类'}.get(task_type, task_type)

        if health_score >= 90:
            return f'[{task_cn}任务] 模型表现卓越 (健康度 {health_score:.0f}/100)。各项指标优秀，已达到生产部署标准。'
        elif health_score >= 75:
            return (
                f'[{task_cn}任务] 模型表现良好 (健康度 {health_score:.0f}/100)。'
                f'弱项维度: {weak_dims_str}。'
                f'按下方路线图微调可预期提升1-3个百分点。'
            )
        elif health_score >= 60:
            return (
                f'[{task_cn}任务] 模型表现一般 (健康度 {health_score:.0f}/100)。'
                f'弱项维度: {weak_dims_str}。'
                f'检测到 {len(issues)} 个可改进问题，建议按下方路线图逐步优化。'
            )
        elif health_score >= 40:
            return (
                f'[{task_cn}任务] 模型表现偏差 (健康度 {health_score:.0f}/100)。'
                f'核心问题: {dim_names.get(worst_dim, "?")} 仅 {worst_score:.0f}/100。'
                f'弱项维度: {weak_dims_str}。需要系统性优化。'
            )
        elif health_score >= 25:
            return (
                f'[{task_cn}任务] 模型表现很差 (健康度 {health_score:.0f}/100)，'
                f'基本不具备实用价值。建议: '
                f'1) 重新检查数据质量 2) 使用RF/GB建立baseline 3) 特征选择去噪。'
            )
        else:
            return (
                f'[{task_cn}任务] 模型几乎无效 (健康度 {health_score:.0f}/100)。'
                f'请确认: 1) 目标列设置正确? 2) 数据包含有效标签? '
                f'3) 数据格式无问题? 建议从LogisticRegression/Ridge baseline开始。'
            )

    # ==================================================================
    # Step 7: 下一步操作 + 调参路线图
    # ==================================================================

    @staticmethod
    def _generate_scientific_next_steps(health_score, sub_scores, issues, suggestions, hyperparams) -> list:
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
            steps.append(
                {
                    'step': 1,
                    'phase': 'Phase 1',
                    'action': '调整学习率 (最重要参数)',
                    'detail': f'收敛质量 {cq:.0f}/100 → 运行LR Finder找到最佳LR范围',
                    'icon': '🔍',
                }
            )

        if of < 75 and not is_cluster:
            steps.append(
                {
                    'step': 2,
                    'phase': 'Phase 4-5',
                    'action': '增强正则化',
                    'detail': f'过拟合风险 {of:.0f}/100 → 依次调整正则化参数'
                    + (
                        ' (weight_decay → dropout)'
                        if algo_lower in ('mlp',)
                        else ' (max_depth → min_samples_split)'
                        if algo_lower.startswith('random_forest')
                        else ' (C/alpha → 简化模型)'
                    ),
                    'icon': '🛡️',
                }
            )

        if ap < 75 or len(issues) > 0:
            steps.append(
                {
                    'step': 3 if steps else 1,
                    'phase': 'Phase 3',
                    'action': '提升模型容量/切换算法',
                    'detail': f'绝对性能 {ap:.0f}/100 → '
                    + (
                        '增加n_estimators或切换GBDT'
                        if algo_lower in ('random_forest', 'random_forest_regressor')
                        else '尝试更深的网络'
                        if algo_lower in ('mlp',)
                        else '尝试不同聚类算法'
                        if is_cluster
                        else '考虑升级到集成方法(RandomForest/GBDT)'
                    ),
                    'icon': '📈',
                }
            )

        # Optuna建议 — 有多个issue时推荐
        if health_score < 85 and len(issues) >= 1:
            steps.append(
                {
                    'step': 4 if steps else 1,
                    'phase': '进阶',
                    'action': '使用Optuna自动搜索',
                    'detail': '对确定的2-3个关键超参数运行Optuna TPE搜索 (50-100 trials)',
                    'icon': '🤖',
                }
            )

        # 良好模型 — 给出部署建议而非优化建议
        if health_score >= 85:
            steps.append(
                {
                    'step': len(steps) + 1,
                    'phase': '部署',
                    'action': '模型已达到生产标准',
                    'detail': '可进行: 模型压缩/量化 → 部署上线 → 持续监控drift',
                    'icon': '✅',
                }
            )

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
        _DL_ONLY = {'mlp', 'transformer_bert'}  # dropout/batch_size/lr仅对DL有意义
        _TREE_ONLY = {
            'random_forest',
            'random_forest_regressor',
            'gradient_boosting',
            'gradient_boosting_regressor',
            'decision_tree',
        }  # n_estimators/max_depth仅对树模型有意义
        _CLUSTER_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}

        is_dl = algo_lower in _DL_ONLY
        is_tree = algo_lower in _TREE_ONLY
        is_cluster = algo_lower in _CLUSTER_ALGOS or task_type == 'clustering'

        # ── 聚类专属路线图 ──
        if is_cluster:
            return [
                {
                    'phase': 1,
                    'param': 'n_clusters',
                    'name': '簇数选择',
                    'method': 'Elbow Method + Silhouette Analysis',
                    'reason': '确定最佳簇数是聚类最重要的决策',
                    'urgency': 'high' if sub_scores.get('absolute_perf', 100) < 70 else 'medium',
                    'note': f'当前silhouette仅{sub_scores.get("absolute_perf", 0):.0f}/100 — 尝试不同k值'
                    if sub_scores.get('absolute_perf', 100) < 70
                    else '通过Elbow曲线找到拐点',
                },
                {
                    'phase': 2,
                    'param': 'algorithm',
                    'name': '聚类算法选择',
                    'method': '对比 K-Means vs DBSCAN vs Agglomerative',
                    'reason': '不同算法假设不同的簇形状 — K-Means球形/DBSCAN任意形/层次树形',
                    'urgency': 'medium',
                    'note': f'当前{algorithm}, 建议对比其他算法',
                },
                {
                    'phase': 3,
                    'param': 'init',
                    'name': '初始化方法',
                    'method': 'k-means++ vs random',
                    'reason': 'k-means++ 提供更好的初始质心，通常收敛到更优解',
                    'urgency': 'low' if algo_lower != 'kmeans' else 'medium',
                },
                {
                    'phase': 4,
                    'param': 'max_iter',
                    'name': '最大迭代次数',
                    'method': '逐步增加至收敛稳定 (100 → 500)',
                    'reason': '复杂数据集需要更多迭代才能收敛',
                    'urgency': 'low',
                },
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
            train_accs = [
                m.get('train_accuracy', m.get('accuracy', 0))
                for m in history
                if m.get('train_accuracy') is not None or m.get('accuracy') is not None
            ]
            test_acc = final_metrics.get('test_accuracy', final_metrics.get('accuracy', 0))
            avg_train = float(np.mean(train_accs)) if train_accs else test_acc
            # 训练和测试都 < 0.55 且 gap < 0.10 → 欠拟合
            gap = abs(avg_train - test_acc) if train_accs else 0
            return avg_train < 0.55 and test_acc < 0.55 and gap < 0.10 and test_acc > 0
        elif task_type == 'regression':
            train_r2s = [
                m.get('train_r2', m.get('r2', 0))
                for m in history
                if m.get('train_r2') is not None or m.get('r2') is not None
            ]
            test_r2 = final_metrics.get('test_r2', final_metrics.get('r2', 0))
            avg_train = float(np.mean(train_r2s)) if train_r2s else test_r2
            gap = abs(avg_train - test_r2) if train_r2s else 0
            return avg_train < 0.25 and test_r2 < 0.25 and gap < 0.10
        return False

    # ==================================================================
    # 智能参数推荐 v3 — 连续缩放 + 数据感知 + 反馈学习
    # ==================================================================
    #
    # v3 核心改进:
    #   1. 连续缩放: 用 log(n_samples) 平滑插值替代 4 桶分档
    #   2. 数据感知: n_estimators/max_depth/dropout 根据 n_samples × n_features
    #      动态计算, 而非写死固定值
    #   3. 反馈学习: recommend_retry_params() 接受历史训练结果,
    #      诊断低分根因(数据问题 vs 参数问题), 给出有意义的参数变化
    #   4. 特征质量: _estimate_signal_quality() 评估预测信号强度,
    #      低信号数据自动加正则化/降模型复杂度
    #   5. 智能 GridSearchCV 建议: 根据性能缺口决定是否推荐调优
    #
    # 设计原则 (Google Tuning Playbook):
    #   - 参数之间不是独立的: n_estimators↑ → lr↓
    #   - 数据量决定模型复杂度上限
    #   - 特征质量决定正则化强度
    #   - 反馈闭环: 低分先诊断根因, 再决定调参 vs 换算法 vs 检查数据

    # ── 算法族分类 (决定参数逻辑) ──
    _TREE_ALGOS = {
        'random_forest',
        'random_forest_regressor',
        'gradient_boosting',
        'gradient_boosting_regressor',
        'decision_tree',
    }
    _LINEAR_ALGOS = {'logistic_regression', 'linear_regression', 'ridge', 'svm', 'svr'}
    _KNN_ALGOS = {'knn', 'knn_regressor'}
    _CLUSTER_ALGOS = {'kmeans', 'dbscan', 'agglomerative', 'minibatch_kmeans'}
    _DL_ALGOS = {'mlp', 'transformer_bert'}

    @staticmethod
    def _continuous_scale(n_samples: int) -> dict:
        """连续缩放函数 — 用 log(n) 平滑插值替代固定分桶

        Returns:
            {epochs, batch_size, learning_rate, dropout, weight_decay, val_size}
            所有值在 n=100 和 n=1e6 之间平滑过渡
        """
        # 对数归一化: log10(n) 映射到 [0, 1] 区间 (100 → 0, 1e6 → 1)
        import math

        log_n = math.log10(max(100, min(n_samples, 1_000_000)))
        t = (log_n - 2.0) / 4.0  # log10(100)=2, log10(1e6)=6 → range 4
        t = max(0.0, min(1.0, t))

        # 平滑插值 (clamp + lerp)
        def lerp(a, b, x):
            return a + (b - a) * x

        return {
            'epochs': round(lerp(35, 4, t)),  # 小数据多轮, 大数据少轮
            'batch_size': round(lerp(12, 160, t) / 8) * 8,  # 量化到 8 的倍数
            'learning_rate': round(lerp(3e-4, 2e-3, t), 6),  # 小数据低lr防过拟合
            'dropout': round(lerp(0.55, 0.15, t), 2),  # 小数据强dropout
            'weight_decay': 10 ** lerp(-2.5, -5.0, t),  # log scale: 3e-3 → 1e-5
            'val_size': round(lerp(0.22, 0.08, t), 2),  # 小数据多留验证
        }

    @staticmethod
    def _estimate_signal_quality(analysis: dict) -> dict:
        """估算数据集预测信号质量

        用多个代理指标推断特征-标签关联强度:
          - avg_correlation: 特征间平均相关性 (高→冗余, 低→独立)
          - high_corr_pairs:  高相关特征对数量
          - n_features / n_samples 比值 (高→易过拟合)
          - missing_rate:      缺失率 (高→噪声)
          - imbalanced:        是否类别不平衡

        Returns:
            {score: 0-1, level: 'strong'|'moderate'|'weak'|'noise',
             issues: [...], regularization_bias: float}

        Raises:
            ValueError: 当 analysis 缺少必需字段 (n_samples, n_features) 时
        """
        # ── 必需字段验证 ──
        if 'n_samples' not in analysis:
            raise ValueError(
                '信号质量评估失败: analysis 缺少必需字段 "n_samples"。'
                '请确认 DatasetAnalyzer.analyze() 的输出包含样本数。'
            )
        if 'n_features' not in analysis:
            raise ValueError(
                '信号质量评估失败: analysis 缺少必需字段 "n_features"。'
                '请确认 DatasetAnalyzer.analyze() 的输出包含特征数。'
            )
        n_samples = max(analysis['n_samples'], 1)
        n_features = analysis['n_features']
        avg_corr = analysis.get('avg_correlation', 0.2)
        high_corr = analysis.get('high_corr_pairs', 0)
        missing = analysis.get('missing_rate', 0)
        imbalanced = analysis.get('imbalanced', False)
        wide = analysis.get('wide_data', False)

        score = 0.70  # 基础分
        issues = []

        # 特征/样本比: >1:10 扣分 (高维小样本)
        ratio = n_features / max(n_samples, 1)
        if ratio > 0.5:  # 特征数 > 样本数的一半
            score -= 0.25
            issues.append(f'特征/样本比过高({ratio:.2f}), 极易过拟合')
        elif ratio > 0.1:
            score -= 0.10
            issues.append(f'特征/样本比偏高({ratio:.2f}), 需强正则化')

        # 宽数据 (特征 > 样本)
        if wide:
            score -= 0.30
            issues.append(f'宽数据: {n_features}特征 > {n_samples}样本, 强烈建议降维')

        # 缺失率
        if missing > 0.20:
            score -= 0.25
            issues.append(f'缺失率过高({missing:.1%}), 数据质量差')
        elif missing > 0.08:
            score -= 0.10
            issues.append(f'缺失率偏高({missing:.1%})')

        # 高相关性 → 多重共线性 → 线性模型不稳定
        if high_corr > 10:
            score -= 0.10
            issues.append(f'{high_corr}对高相关特征, 存在多重共线性')
        elif high_corr > 3:
            score -= 0.05

        # 类别不平衡
        if imbalanced:
            score -= 0.08
            issues.append('类别不平衡, 需class_weight或重采样')

        # avg_correlation 适度(0.15-0.35)加分 (说明特征间有共享信息)
        if 0.12 <= avg_corr <= 0.35:
            score += 0.08

        score = max(0.05, min(0.98, score))

        if score >= 0.70:
            level = 'strong'
        elif score >= 0.45:
            level = 'moderate'
        elif score >= 0.25:
            level = 'weak'
        else:
            level = 'noise'

        # 正则化偏置: 信号越弱, 正则化越强
        regularization_bias = 1.0 + (1.0 - score) * 3.0  # 1.0 ~ 4.0

        return {
            'score': round(score, 3),
            'level': level,
            'issues': issues,
            'ratio': round(ratio, 4),
            'regularization_bias': round(regularization_bias, 2),
        }

    @staticmethod
    def _generate_tree_params(algorithm: str, n_samples: int, n_features: int, signal: dict, scale: dict) -> dict:
        """为树模型生成数据感知参数

        n_estimators:
          - 基础值 = 50 + 50 * log10(n_samples)
          - 信号弱 → +30% 树数量 (更多树平均降噪)
          - 特征多 → +20% 树数量 (更多特征需要更多树)

        max_depth:
          - 信号强 → None (让树充分生长)
          - 信号中等 → clip(n_features * 0.6, 8, 25)
          - 信号弱 → clip(n_features * 0.3, 3, 10)

        min_samples_split/min_samples_leaf:
          - 信号弱 → 增大 (强正则化)
        """
        import math

        log_n = math.log10(max(100, n_samples))

        # n_estimators: 平滑增长, 不是固定 200
        base_trees = int(40 + 45 * (log_n - 2.0))  # n=100→40, n=10K→130, n=100K→175
        base_trees = max(40, min(300, base_trees))

        # 信号调整
        sig_factor = 1.0
        if signal['level'] == 'weak':
            sig_factor = 1.35  # 弱信号: 更多树 + bagging 降噪
        elif signal['level'] == 'noise':
            sig_factor = 1.50
        elif signal['level'] == 'strong':
            sig_factor = 0.85  # 强信号: 少树足够

        # 特征数调整
        feat_factor = 1.0 + max(0, (n_features - 15) / 100)  # 特征>15时逐步增加

        n_estimators = int(base_trees * sig_factor * feat_factor)
        n_estimators = max(30, min(500, n_estimators))

        # max_depth: 信号感知
        if signal['level'] == 'strong':
            max_depth = None  # 不限制
            min_samples_leaf = 1
            min_samples_split = 2
        elif signal['level'] == 'moderate':
            max_depth = max(6, min(25, int(n_features * 0.6)))
            min_samples_leaf = max(1, int(n_samples**0.25 / 3))
            min_samples_split = max(2, min_samples_leaf * 2)
        elif signal['level'] == 'weak':
            max_depth = max(3, min(12, int(n_features * 0.3)))
            min_samples_leaf = max(2, int(n_samples**0.25 / 2))
            min_samples_split = max(4, min_samples_leaf * 3)
        else:  # noise
            max_depth = max(2, min(6, int(n_features * 0.15)))
            min_samples_leaf = max(5, int(n_samples**0.3 / 2))
            min_samples_split = max(8, min_samples_leaf * 2)

        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
        }

        # GBDT 特有参数
        if algorithm.startswith('gradient'):
            gb_lr = round(0.2 / math.sqrt(max(1, n_estimators / 50)), 4)
            gb_lr = max(0.01, min(0.3, gb_lr))
            params['learning_rate'] = gb_lr
            params['subsample'] = 0.7 if signal['level'] in ('weak', 'noise') else 0.8
            if n_estimators > 150:
                params['learning_rate'] = round(gb_lr * 0.6, 4)
                n_estimators = int(n_estimators * 1.3)
                params['n_estimators'] = min(500, n_estimators)
        else:
            params['max_features'] = 'sqrt' if n_features > 15 else None

        return params

    @staticmethod
    def _generate_linear_params(algorithm: str, n_samples: int, n_features: int, signal: dict) -> dict:
        """为线性模型生成数据感知参数

        C (SVM/LR) 或 alpha (Ridge):
          - 信号强 → C偏大(弱正则化), alpha偏小
          - 信号弱 → C偏小(强正则化), alpha偏大
        """

        reg_bias = signal['regularization_bias']
        signal['ratio']

        params = {'max_iter': 3000}

        if algorithm in ('svm', 'svr'):
            # C: 信号越弱 → C越小 (强正则化)
            base_c = 1.0
            params['C'] = round(base_c / reg_bias, 4)
            params['kernel'] = 'rbf'
            params['gamma'] = 'scale'
            # 高维 + 弱信号 → 尝试 linear kernel
            if n_features > 50 and signal['level'] in ('weak', 'noise'):
                params['kernel'] = 'linear'
        elif algorithm == 'ridge':
            # alpha: 信号越弱 → alpha越大 (强正则化)
            params['alpha'] = round(1.0 * reg_bias, 2)
        elif algorithm == 'logistic_regression':
            params['C'] = round(1.0 / reg_bias, 4)
            params['penalty'] = 'l2'
            params['solver'] = 'lbfgs'
            # 小样本用 liblinear (更稳定)
            if n_samples < 500:
                params['solver'] = 'liblinear'
        elif algorithm == 'linear_regression':
            params['fit_intercept'] = True

        return params

    @staticmethod
    def _generate_knn_params(n_samples: int, n_features: int, signal: dict) -> dict:
        """为 KNN 生成数据感知参数

        n_neighbors = clip(sqrt(n_samples) * signal_factor, 3, 30)
        信号弱 → 更多邻居 (平滑降噪)
        """
        import math

        base_k = int(math.sqrt(n_samples))
        if signal['level'] == 'strong':
            k = max(3, base_k // 2)
        elif signal['level'] == 'moderate':
            k = max(3, base_k)
        else:
            k = max(5, base_k * 2)

        k = max(3, min(31, k))
        if k % 2 == 0:
            k += 1  # 奇数防平票

        return {
            'n_neighbors': k,
            'weights': 'distance',
            'metric': 'minkowski',
        }

    @staticmethod
    def _generate_dl_params(algorithm: str, n_samples: int, n_features: int, signal: dict, scale: dict) -> dict:
        """为 DL 模型生成数据感知参数

        hidden_layers:
          - 层数 = clip(1 + log2(n_features/10), 1, 5)
          - 每层宽度 = n_features * multiplier → 逐层衰减 0.5x
          - 信号弱 → 更窄的网络 (防过拟合)
        """
        import math

        if algorithm == 'transformer_bert':
            return {
                'learning_rate': 2e-5,
                'batch_size': 16,
                'max_length': 256,
                'epochs': scale['epochs'],
                'warmup_steps': 500,
                'dropout': 0.1,
            }

        # MLP: 层数 = f(n_features)
        n_layers = max(1, min(5, int(1 + math.log2(max(1, n_features / 10)))))
        # 第一层宽度
        if signal['level'] == 'strong':
            first_width = int(n_features * 4)
        elif signal['level'] == 'moderate':
            first_width = int(n_features * 2.5)
        elif signal['level'] == 'weak':
            first_width = int(n_features * 1.5)
        else:
            first_width = int(n_features * 1.0)

        first_width = max(16, min(1024, first_width))
        # 量化到 2 的幂附近
        import math as _m

        first_width = 2 ** int(_m.log2(first_width))

        hidden = []
        w = first_width
        for _ in range(n_layers):
            hidden.append(w)
            w = max(8, w // 2)

        return {
            'hidden_layers': hidden,
            'learning_rate': scale['learning_rate'],
            'dropout': scale['dropout'],
            'batch_size': scale['batch_size'],
            'weight_decay': scale['weight_decay'],
            'early_stopping_patience': scale['epochs'] // 2,
            'val_size': scale['val_size'],
        }

    # ── 公共 API ──

    @staticmethod
    def recommend_initial_params(
        analysis: dict, algorithm: str, ml_task_type: str = 'classification', framework: str = 'sklearn'
    ) -> dict:
        """训练前智能参数推荐 v3 — 连续缩放 + 数据感知

        Args:
            analysis:   DatasetAnalyzer.analyze() 的输出
            algorithm:  算法名 (随机森林/gradient_boosting/...)
            ml_task_type: classification/regression/clustering
            framework:  sklearn/pytorch/transformers

        Returns:
            {params, reason, signal_quality, tips, gridsearch_suggestion, ...}
        """
        n_samples = analysis.get('n_samples', 1000)
        n_features = analysis.get('n_features', 10)
        imbalanced = analysis.get('imbalanced', False)
        high_dim = analysis.get('high_dim', False)
        missing_rate = analysis.get('missing_rate', 0)
        analysis.get('n_classes', 0)

        # ── 1. 连续缩放 ──
        scale = ParameterGuidanceService._continuous_scale(n_samples)

        # ── 2. 信号质量评估 ──
        signal = ParameterGuidanceService._estimate_signal_quality(analysis)

        # ── 3. 算法感知参数生成 ──
        tips = []
        algo_lower = algorithm.lower()
        if algo_lower in ParameterGuidanceService._TREE_ALGOS:
            params = ParameterGuidanceService._generate_tree_params(algorithm, n_samples, n_features, signal, scale)
            algo_family = '集成树'
        elif algo_lower in ParameterGuidanceService._LINEAR_ALGOS:
            params = ParameterGuidanceService._generate_linear_params(algorithm, n_samples, n_features, signal)
            algo_family = '线性模型'
        elif algo_lower in ParameterGuidanceService._KNN_ALGOS:
            params = ParameterGuidanceService._generate_knn_params(n_samples, n_features, signal)
            algo_family = 'KNN'
        elif algo_lower in ParameterGuidanceService._DL_ALGOS:
            params = ParameterGuidanceService._generate_dl_params(algorithm, n_samples, n_features, signal, scale)
            algo_family = '深度学习'
        elif algo_lower in ParameterGuidanceService._CLUSTER_ALGOS or ml_task_type == 'clustering':
            # 聚类算法 — 使用 sqrt(n/2) 经验公式, 而非 n_classes (违反无监督假设)
            import math

            est_k = max(2, min(15, int(math.sqrt(n_samples / 2))))
            params = {
                'n_clusters': est_k,
                'max_iter': 500,
            }
            algo_family = '聚类'
            tips.insert(
                0,
                f'聚类初始化: n_clusters={est_k} (经验公式 sqrt(n/2)), '
                f'请通过 Elbow Method + Silhouette Analysis 确认最佳簇数',
            )
        else:
            # 完全未知算法 — 最小合理默认值
            params = {
                'max_iter': 500,
            }
            algo_family = '未知算法类型'

        # ── 4. 通用参数叠加 ──
        if algorithm not in ('transformer_bert',):
            for k in ('epochs', 'batch_size', 'val_size'):
                if k not in params and k in scale:
                    params[k] = scale[k]

        # 注册 test_size
        if n_samples > 20000:
            params['test_size'] = 0.10
        elif n_samples > 5000:
            params['test_size'] = 0.20
        else:
            params['test_size'] = 0.25

        # ── 5. 特殊场景 & tips ──
        tips = list(signal['issues'])  # 信号质量问题作为 tips

        if imbalanced and ml_task_type == 'classification':
            params['class_weight'] = 'balanced'
            tips.insert(0, '类别不平衡: 已启用 class_weight="balanced"')

        if high_dim and n_features > 100:
            tips.insert(0, f'高维数据({n_features}维): 建议PCA降维或特征选择')

        if missing_rate > 0.05:
            tips.insert(0, f'缺失率{missing_rate:.1%}: 建议先做缺失值处理')

        # ── 6. 构建推荐理由 ──
        signal_cn = {'strong': '强', 'moderate': '中等', 'weak': '弱', 'noise': '极弱'}
        scale_label = '小' if n_samples < 2000 else ('中' if n_samples < 15000 else '大')

        reason_parts = [
            f'{algorithm} ({algo_family})',
            f'{scale_label}规模数据 ({n_samples:,}样本 × {n_features}特征)',
            f'信号质量: {signal_cn.get(signal["level"], "?")} ({signal["score"]:.2f})',
        ]
        if 'n_estimators' in params:
            reason_parts.append(f'动态n_estimators={params["n_estimators"]}')
        reason = ' — '.join(reason_parts) + '。'

        # ── 7. GridSearchCV 建议 (v3: 更智能) ──
        sklearn_algos = (
            ParameterGuidanceService._TREE_ALGOS
            | ParameterGuidanceService._LINEAR_ALGOS
            | ParameterGuidanceService._KNN_ALGOS
        )
        # 仅在中等信号 + 中等数据量时强烈推荐 GridSearchCV
        # 强信号 → 当前参数已足够, 弱信号 → 调参帮助有限, 应换算法或检查数据
        if algorithm in sklearn_algos and 800 < n_samples < 80000:
            if signal['level'] == 'moderate':
                gs_suggest = True
                gridsearch_reason = (
                    f'信号质量中等({signal["score"]:.2f}), GridSearchCV 有较大提升空间。'
                    f'{n_samples:,}样本适合参数搜索, 预计{_estimate_tuning_time(n_samples, len(params), algorithm)}。'
                )
            elif signal['level'] == 'weak':
                gs_suggest = True
                gridsearch_reason = (
                    f'⚠ 信号质量偏低({signal["score"]:.2f}), GridSearchCV 提升有限。'
                    f'强烈建议先检查特征质量/目标列正确性/数据泄漏。'
                    f'如确认数据无误, 可尝试调优但预期提升 <5%。'
                )
            elif signal['level'] == 'strong':
                gs_suggest = False
                gridsearch_reason = (
                    f'信号质量强({signal["score"]:.2f}), 当前参数已接近最优。'
                    f'手动微调关键参数 (n_estimators/learning_rate) 即可, 无需完整GridSearch。'
                )
            else:
                gs_suggest = False
                gridsearch_reason = (
                    f'⚠ 信号极弱({signal["score"]:.2f}), 调参不会显著改善。'
                    f'请优先检查: 1) 目标列是否正确 2) 特征是否包含预测信息 3) 是否存在数据泄漏。'
                )
        else:
            gs_suggest = False
            if n_samples <= 800:
                gridsearch_reason = f'样本量过小({n_samples}), CV分折不稳定, 建议手动调参。'
            elif n_samples >= 80000:
                gridsearch_reason = f'大数据集({n_samples:,}), 单次GridSearch耗时过长, 建议采样后调优。'
            else:
                gridsearch_reason = f'{algorithm} 不支持 sklearn GridSearchCV 调优。'

        # ── 8. 置信度 ──
        if signal['level'] == 'strong':
            confidence = 0.90
        elif signal['level'] == 'moderate':
            confidence = 0.78
        elif signal['level'] == 'weak':
            confidence = 0.60
        else:
            confidence = 0.40  # 低信号 → 告诉用户别期望太高

        return {
            'params': params,
            'reason': reason,
            'signal_quality': signal,
            'scale': scale_label,
            'confidence': round(confidence, 2),
            'tips': tips,
            'gridsearch_suggestion': gs_suggest,
            'gridsearch_reason': gridsearch_reason,
        }

    @staticmethod
    def recommend_retry_params(
        analysis: dict,
        algorithm: str,
        previous_metrics: dict = None,
        previous_params: dict = None,
        ml_task_type: str = 'classification',
        framework: str = 'sklearn',
    ) -> dict:
        """反馈学习: 根据上一次训练结果生成改进参数

        这是 v3 的核心新增功能 — 解决"重复推荐同一套参数"的问题。

        Args:
            analysis:          数据集分析结果
            algorithm:         算法名
            previous_metrics:  上次训练的 final_metrics (e.g. {test_accuracy: 0.65, ...})
            previous_params:   上次使用的超参数 (用于避免重复推荐)
            ml_task_type:      任务类型
            framework:         框架

        Returns:
            与 recommend_initial_params 相同结构, 但参数基于反馈调整
        """
        # 先获取基础推荐
        result = ParameterGuidanceService.recommend_initial_params(analysis, algorithm, ml_task_type, framework)
        params = result['params']
        tips = result['tips'][:]

        if not previous_metrics:
            result['tips'] = tips
            return result

        # ── 诊断: 区分数据问题 vs 参数问题 ──
        signal = result['signal_quality']

        # 获取关键指标
        if ml_task_type == 'regression':
            test_score = previous_metrics.get('test_r2', previous_metrics.get('r2', 0))
            train_score = previous_metrics.get('train_r2', test_score)
            score_name = 'R²'
            low_threshold = 0.30
        elif ml_task_type == 'clustering':
            test_score = previous_metrics.get('test_silhouette_score', previous_metrics.get('silhouette_score', 0))
            train_score = test_score
            score_name = 'Silhouette'
            low_threshold = 0.25
        else:
            test_score = previous_metrics.get('test_accuracy', previous_metrics.get('accuracy', 0))
            train_score = previous_metrics.get('train_accuracy', previous_metrics.get('accuracy', test_score))
            score_name = 'Accuracy'
            low_threshold = 0.55

        gap = abs(train_score - test_score) if train_score else 0
        is_low = test_score < low_threshold
        is_overfitting = gap > 0.10

        # ── 决策树: 低分根因分析 ──
        diagnosis = []
        param_changes = {}

        if is_low and signal['level'] in ('weak', 'noise'):
            # 信号弱 + 评分低 → 数据问题占主导
            diagnosis.append(
                {
                    'type': 'data_issue',
                    'severity': 'high',
                    'message': (
                        f'信号质量{signal["level"]}({signal["score"]:.2f}) + '
                        f'{score_name}={test_score:.3f}偏低 → 数据质量可能是根因。'
                        f'调参提升空间有限 (<5%), 建议优先检查数据。'
                    ),
                    'suggestion': '检查目标列是否正确、特征是否包含预测信息、是否存在数据泄漏',
                }
            )
            # 数据问题: 不要大幅调参, 只小幅加强正则化
            if 'n_estimators' in params:
                params['n_estimators'] = min(500, int(params['n_estimators'] * 1.15))
            if 'max_depth' in params and params['max_depth'] is not None:
                params['max_depth'] = max(3, int(params['max_depth'] * 0.7))
            if 'dropout' in params:
                params['dropout'] = min(0.7, round(params['dropout'] + 0.10, 2))
        elif is_low:
            # 信号尚可但评分低 → 参数/算法问题
            algo_lower = algorithm.lower()
            if algo_lower in ParameterGuidanceService._TREE_ALGOS:
                # 树模型: 增加容量
                if 'n_estimators' in params:
                    params['n_estimators'] = min(500, int(params['n_estimators'] * 1.5))
                    param_changes['n_estimators'] = params['n_estimators']
                if algo_lower.startswith('gradient') and 'learning_rate' in params:
                    params['learning_rate'] = round(params['learning_rate'] * 0.5, 5)
                    n_est = params.get('n_estimators', 250)
                    params['n_estimators'] = min(500, int(n_est * 1.5))
                    param_changes['lr×0.5 + n_est×1.5'] = True
                diagnosis.append(
                    {
                        'type': 'underfit',
                        'severity': 'medium',
                        'message': (
                            f'{score_name}={test_score:.3f}偏低 + 信号{signal["level"]} → '
                            f'增加模型容量 (n_estimators↑, lr↓)。'
                            f'如仍无改善, 建议切换算法 (RF→GB 或反之)。'
                        ),
                        'suggestion': '增加树数量+降低学习率; 或切换到更强的算法',
                    }
                )
            elif algo_lower in ParameterGuidanceService._LINEAR_ALGOS:
                # 线性模型评分低 → 可能是非线性关系
                diagnosis.append(
                    {
                        'type': 'algorithm_mismatch',
                        'severity': 'high',
                        'message': (
                            f'线性模型 {score_name}={test_score:.3f} → 数据可能存在非线性关系, 线性模型容量不足。'
                        ),
                        'suggestion': '强烈建议切换到 RandomForest 或 GradientBoosting',
                    }
                )
                tips.insert(0, '⚠ 线性模型评分低, 建议切换到集成树模型 (RF/GBDT)')
            elif algo_lower in ParameterGuidanceService._DL_ALGOS:
                diagnosis.append(
                    {
                        'type': 'dl_underfit',
                        'severity': 'medium',
                        'message': f'深度学习 {score_name}={test_score:.3f}偏低 → 可能需要更多epochs或调整架构。',
                        'suggestion': '增加 epochs + 调整 hidden_layers 宽度',
                    }
                )
                if 'epochs' in params:
                    params['epochs'] = min(100, params['epochs'] * 2)
                if 'hidden_layers' in params:
                    wider = [min(1024, w * 2) for w in params['hidden_layers']]
                    params['hidden_layers'] = wider
        elif is_overfitting:
            # 过拟合: 增强正则化
            diagnosis.append(
                {
                    'type': 'overfitting',
                    'severity': 'medium',
                    'message': (
                        f'Train-Test gap={gap:.3f} → 过拟合。加强正则化: max_depth↓, dropout↑, weight_decay↑。'
                    ),
                    'suggestion': '限制max_depth, 增加dropout/weight_decay, 或增大test_size',
                }
            )
            if 'max_depth' in params and params['max_depth'] is not None:
                params['max_depth'] = max(3, params['max_depth'] // 2)
            elif 'max_depth' in params and params['max_depth'] is None:
                params['max_depth'] = max(5, int(analysis.get('n_features', 10) * 0.4))
            if 'dropout' in params:
                params['dropout'] = min(0.75, round(params['dropout'] + 0.15, 2))
            if 'weight_decay' in params:
                params['weight_decay'] = round(params['weight_decay'] * 3, 6)
            if 'min_samples_leaf' in params:
                params['min_samples_leaf'] = max(2, params['min_samples_leaf'] * 3)
        else:
            # 评分OK → 微调建议
            diagnosis.append(
                {
                    'type': 'fine_tune',
                    'severity': 'low',
                    'message': (f'{score_name}={test_score:.3f} 已达可用水平。小幅微调关键参数可提升 1-3%。'),
                    'suggestion': '用 GridSearchCV 在小范围搜索 n_estimators/learning_rate',
                }
            )
            if 'n_estimators' in params:
                params['n_estimators'] = min(500, int(params['n_estimators'] * 1.15))

        # ── 避免与上次完全相同的参数 ──
        if previous_params:
            same_keys = 0
            total_keys = 0
            for k in params:
                if k in previous_params and k not in (
                    'test_size',
                    'val_size',
                    'framework',
                    'algorithm',
                    'ml_task_type',
                ):
                    total_keys += 1
                    if params[k] == previous_params[k]:
                        same_keys += 1
            if total_keys > 0 and same_keys / total_keys > 0.85:
                # 参数几乎没变 → 强制微调
                if 'n_estimators' in params:
                    params['n_estimators'] = min(500, int(params['n_estimators'] * 1.2))
                if 'learning_rate' in params:
                    params['learning_rate'] = round(params['learning_rate'] * 0.7, 5)
                tips.insert(0, '检测到参数与上次几乎相同, 已自动调整以避免重复。')

        result['params'] = params
        result['diagnosis'] = diagnosis
        result['tips'] = tips
        result['feedback_applied'] = True

        return result

    @staticmethod
    def diagnose_low_score(analysis: dict, algorithm: str, metrics: dict, ml_task_type: str = 'classification') -> dict:
        """独立诊断: 低分根因分析 (供前端 "AI诊断" Tab 使用)

        Returns:
            {root_cause: 'data'|'algorithm'|'parameter'|'unknown',
             confidence: float,
             explanation: str,
             suggested_actions: [...]}
        """
        signal = ParameterGuidanceService._estimate_signal_quality(analysis)

        if ml_task_type == 'regression':
            score = metrics.get('test_r2', metrics.get('r2', 0))
            train_score = metrics.get('train_r2', score)
            low_threshold = 0.30
            score_name = 'R²'
        elif ml_task_type == 'clustering':
            score = metrics.get('test_silhouette_score', metrics.get('silhouette_score', 0))
            train_score = score
            low_threshold = 0.25
            score_name = 'Silhouette'
        else:
            score = metrics.get('test_accuracy', metrics.get('accuracy', 0))
            train_score = metrics.get('train_accuracy', metrics.get('accuracy', score))
            low_threshold = 0.55
            score_name = 'Accuracy'

        gap = abs(train_score - score)
        actions = []

        # 决策逻辑
        if signal['level'] in ('weak', 'noise') and score < low_threshold:
            root_cause = 'data'
            confidence = 0.75 + (1.0 - signal['score']) * 0.2
            explanation = (
                f'信号质量{signal["level"]}({signal["score"]:.2f}) + '
                f'{score_name}={score:.3f} → 数据质量是主因。'
                f'特征预测力不足, 调参空间有限。'
            )
            actions = [
                '检查目标列 (target_column) 是否正确设置了',
                '验证特征与目标的相关性 — 用 df.corr()[target] 排序查看',
                '确认无数据泄漏 (如 target 的线性组合出现在特征中)',
                '尝试特征工程: 交互特征、多项式特征、降维',
                '如数据无误, 接受当前分数作为该数据集的性能上限',
            ]
        elif score < low_threshold * 0.6:
            root_cause = 'algorithm'
            confidence = 0.65
            explanation = (
                f'{score_name}={score:.3f} 极低, 但信号{signal["level"]}。当前算法 ({algorithm}) 可能不匹配数据分布。'
            )
            algo_lower = algorithm.lower()
            if algo_lower in ParameterGuidanceService._LINEAR_ALGOS:
                actions = [
                    '线性模型无法捕获非线性关系 → 切换到 RandomForest 或 GradientBoosting',
                    '如果是分类问题, 尝试 SVM(rbf kernel) 或 MLP',
                ]
            elif algo_lower in ParameterGuidanceService._TREE_ALGOS:
                actions = [
                    '尝试切换到不同树模型: RF ↔ GBDT',
                    '增加 n_estimators 到 300-500',
                    '如果样本<500, 树模型可能过拟合 → 尝试 SVM 或 逻辑回归',
                ]
            else:
                actions = [
                    '尝试切换到 RandomForest (最鲁棒的baseline)',
                    '用 AutoML 对比多种算法',
                ]
        elif gap > 0.15:
            root_cause = 'parameter'
            confidence = 0.72
            explanation = f'Train-Test gap={gap:.3f} → 严重过拟合。正则化参数需要调整。'
            actions = [
                '限制 max_depth (树模型) 或增加 dropout/weight_decay (DL)',
                '增大 test_size 以获得更可靠的泛化估计',
                '对树模型: 增大 min_samples_split 和 min_samples_leaf',
                '考虑使用交叉验证 (CV=5) 评估真实泛化性能',
            ]
        elif score < low_threshold:
            root_cause = 'parameter'
            confidence = 0.55
            explanation = f'{score_name}={score:.3f} 偏低, 但信号{signal["level"]}。可能是参数未达最优。'
            actions = [
                '使用 GridSearchCV 在关键参数上搜索',
                f'当前算法({algorithm}): 重点调 n_estimators 和 learning_rate',
                '如果调参后分数仍不提升 → 根因可能是数据或算法, 回看上方建议',
            ]
        else:
            root_cause = 'unknown'
            confidence = 0.40
            explanation = f'{score_name}={score:.3f}, 各项指标正常。'
            actions = ['分数已在合理范围, 如需提升请用 GridSearchCV 微调']

        return {
            'root_cause': root_cause,
            'confidence': round(min(0.95, confidence), 2),
            'explanation': explanation,
            'suggested_actions': actions,
            'signal_quality': signal,
            'metrics_summary': {
                'score': score,
                'score_name': score_name,
                'train_score': train_score,
                'gap': round(gap, 4),
                'is_low': score < low_threshold,
                'is_overfitting': gap > 0.10,
            },
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
