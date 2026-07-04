"""
独立测试数据生成器
当无法从网络获取真实独立测试数据时, 使用合成方法生成测试集

策略:
- 分类: 用不同的随机种子从训练分布采样, 加小幅度高斯噪声模拟分布偏移
- 回归: 用高斯Copula拟合训练分布后采样
- 聚类: 用高斯混合模型(GMM)拟合后采样
- NLP: 用text_augment做同义词替换+字符扰动

所有生成使用 seed=999 (训练时用seed=42), 确保可复现且独立于训练切分
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# 独立于训练时的随机种子 (训练用42)
INDEPENDENT_SEED = 999


class TestDataGenerator:
    """生成独立测试数据的工具类"""

    # ---- 主入口 ----

    @staticmethod
    def from_training_data(df: pd.DataFrame, target_col: str, task_type: str,
                           n_samples: int = None, perturbation: float = 0.05) -> pd.DataFrame:
        """
        从训练数据分布生成独立测试集

        Args:
            df: 训练数据DataFrame
            target_col: 目标列名
            task_type: 'classification', 'regression', 'clustering', or 'nlp'
            n_samples: 生成样本数 (默认取min(原数据行数, 500))
            perturbation: 噪声幅度 (0.0~1.0, 越大分布偏移越多)

        Returns:
            生成的测试DataFrame (含目标列)
        """
        rng = np.random.RandomState(INDEPENDENT_SEED)

        if n_samples is None:
            n_samples = min(len(df), 500)

        X = df.drop(columns=[target_col]) if target_col in df.columns else df
        y = df[target_col] if target_col in df.columns else None

        if task_type == 'classification':
            return TestDataGenerator._for_classification(X, y, target_col, n_samples, perturbation, rng)
        elif task_type == 'regression':
            return TestDataGenerator._for_regression(X, y, target_col, n_samples, perturbation, rng)
        elif task_type == 'clustering':
            return TestDataGenerator._for_clustering(X, target_col, n_samples, perturbation, rng)
        elif task_type == 'nlp':
            return TestDataGenerator._for_nlp(df, target_col, n_samples, rng)
        else:
            # 默认: 分类处理
            return TestDataGenerator._for_classification(X, y, target_col, n_samples, perturbation, rng)

    # ---- 分类任务 ----

    @staticmethod
    def _for_classification(X, y, target_col, n_samples, perturbation, rng):
        """
        分类任务合成:
        1. 为每个类别用SMOTE-like插值生成新样本
        2. 加小幅度高斯噪声 (模拟自然分布偏移)
        """
        from sklearn.preprocessing import LabelEncoder

        # 编码标签
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        classes = np.unique(y_enc)
        n_classes = len(classes)
        samples_per_class = max(1, n_samples // n_classes)

        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])

        generated_rows = []
        generated_labels = []

        for cls in classes:
            cls_mask = y_enc == cls
            cls_X_num = X_num[cls_mask].values
            cls_count = len(cls_X_num)

            if cls_count == 0:
                continue

            for _ in range(samples_per_class):
                # 随机选两个同类样本做插值
                if cls_count >= 2:
                    i, j = rng.choice(cls_count, size=2, replace=False)
                    alpha = rng.uniform(-0.2, 1.2)  # 允许小幅外推
                    new_num = cls_X_num[i] + alpha * (cls_X_num[j] - cls_X_num[i])
                else:
                    new_num = cls_X_num[0].copy()

                # 加高斯噪声 (每特征独立)
                std = np.std(cls_X_num, axis=0) + 1e-8
                noise = rng.normal(0, perturbation * std, size=len(new_num))
                new_num += noise

                generated_rows.append(new_num)
                generated_labels.append(cls)

        if not generated_rows:
            # 回退: 直接复制+噪声
            n = min(n_samples, len(X))
            indices = rng.choice(len(X), size=n, replace=False)
            X_sample = X_num.iloc[indices].values
            std = np.std(X_sample, axis=0) + 1e-8
            noise = rng.normal(0, perturbation * std, size=X_sample.shape)
            X_sample += noise
            generated_rows = X_sample
            generated_labels = y_enc[indices]

        gen_X = pd.DataFrame(generated_rows, columns=X_num.columns)

        # 分类特征: 从原分布随机采样
        if not X_cat.empty:
            cat_indices = rng.choice(len(X_cat), size=len(gen_X), replace=True)
            gen_X_cat = X_cat.iloc[cat_indices].reset_index(drop=True)
            gen_X = pd.concat([gen_X.reset_index(drop=True), gen_X_cat], axis=1)

        # 确保列顺序与原数据一致
        gen_X = gen_X[X.columns]

        gen_y = le.inverse_transform(np.array(generated_labels))
        result = gen_X.copy()
        result[target_col] = gen_y

        logger.info(f"Generate classification test: {len(result)} samples, {n_classes} classes")
        return result

    # ---- 回归任务 ----

    @staticmethod
    def _for_regression(X, y, target_col, n_samples, perturbation, rng):
        """
        回归任务合成:
        用多元高斯分布 (Copula) 拟合特征+目标联合分布, 重新采样
        """
        X_num = X.select_dtypes(include=[np.number])
        X_cat = X.select_dtypes(exclude=[np.number])

        data = X_num.copy()
        data[target_col] = y.values if hasattr(y, 'values') else y

        # 标准化
        mean = data.mean()
        std = data.std() + 1e-8
        data_std = (data - mean) / std

        # 用经验协方差矩阵采样
        cov = np.cov(data_std.values.T)
        try:
            generated = rng.multivariate_normal(
                np.zeros(len(mean)), cov, size=n_samples
            )
        except np.linalg.LinAlgError:
            # 协方差矩阵奇异, 逐列采样
            generated = np.zeros((n_samples, len(mean)))
            for i in range(len(mean)):
                generated[:, i] = rng.normal(0, np.sqrt(max(cov[i, i], 1e-8)), size=n_samples)

        # 加噪声后还原
        generated += rng.normal(0, perturbation, size=generated.shape)
        generated = generated * std.values + mean.values

        gen_df = pd.DataFrame(generated, columns=data.columns)

        # 分类特征随机采样
        if not X_cat.empty:
            cat_indices = rng.choice(len(X_cat), size=n_samples, replace=True)
            gen_cat = X_cat.iloc[cat_indices].reset_index(drop=True)
            gen_df = pd.concat([gen_df.reset_index(drop=True), gen_cat], axis=1)

        # 拆分特征和目标
        result = gen_df[X.columns].copy()
        result[target_col] = gen_df[target_col]

        logger.info(f"Generate regression test: {len(result)} samples")
        return result

    # ---- 聚类任务 ----

    @staticmethod
    def _for_clustering(X, target_col, n_samples, perturbation, rng):
        """
        聚类任务合成:
        用高斯混合模型(GMM)拟合数据分布后采样
        """
        try:
            from sklearn.mixture import GaussianMixture
            X_num = X.select_dtypes(include=[np.number])

            n_components = min(8, max(2, n_samples // 50))
            gmm = GaussianMixture(n_components=n_components, random_state=INDEPENDENT_SEED)
            gmm.fit(X_num.values)

            generated, _ = gmm.sample(n_samples)
            # 加噪声
            std = np.std(X_num.values, axis=0) + 1e-8
            generated += rng.normal(0, perturbation * std, size=generated.shape)

            gen_df = pd.DataFrame(generated, columns=X_num.columns)

            # 分类特征随机采样
            X_cat = X.select_dtypes(exclude=[np.number])
            if not X_cat.empty:
                cat_indices = rng.choice(len(X_cat), size=n_samples, replace=True)
                gen_cat = X_cat.iloc[cat_indices].reset_index(drop=True)
                gen_df = pd.concat([gen_df.reset_index(drop=True), gen_cat], axis=1)

            gen_df = gen_df[X.columns]

            # 聚类无真实标签: 用聚类结果作为伪标签
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_components, random_state=INDEPENDENT_SEED, n_init=10)
            gen_df[target_col] = km.fit_predict(gen_df.values)

            logger.info(f"Generate clustering test: {len(gen_df)} samples")
            return gen_df

        except Exception as e:
            logger.warning(f"GMM generation failed ({e}), falling back to bootstrap")
            indices = rng.choice(len(X), size=min(n_samples, len(X)), replace=True)
            return X.iloc[indices].assign(**{target_col: 0}).reset_index(drop=True)

    # ---- NLP 任务 ----

    @staticmethod
    def _for_nlp(df, target_col, n_samples, rng):
        """
        NLP任务合成:
        用 text_augment.py 的同义词替换+字符扰动生成变体
        """
        text_col = None
        for col in df.columns:
            if col != target_col:
                text_col = col
                break

        if text_col is None:
            # 单列数据
            logger.warning("NLP: no text column found, returning bootstrap sample")
            indices = rng.choice(len(df), size=min(n_samples, len(df)), replace=True)
            return df.iloc[indices].reset_index(drop=True)

        try:
            from app.utils.text_augment import augment_texts

            n = min(n_samples, len(df))
            indices = rng.choice(len(df), size=n, replace=False)

            original_texts = df[text_col].iloc[indices].tolist()
            labels = df[target_col].iloc[indices].values

            # 用不同的增强策略 (seed 999 vs 训练时的默认)
            augmented = augment_texts(original_texts, augment_factor=2, seed=INDEPENDENT_SEED)

            # 每个原文本取第一个增强结果
            gen_texts = []
            gen_labels = []
            for i, aug_list in enumerate(augmented):
                if aug_list:
                    gen_texts.append(aug_list[0])  # 取第一个增强版
                    gen_labels.append(labels[i])

            result = pd.DataFrame({
                text_col: gen_texts,
                target_col: gen_labels
            })

            logger.info(f"Generate NLP test: {len(result)} samples (augmented)")
            return result

        except Exception as e:
            logger.warning(f"NLP augmentation failed ({e}), falling back to bootstrap")
            indices = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
            return df.iloc[indices].reset_index(drop=True)

    # ---- 验证工具 ----

    @staticmethod
    def validate_test_data(test_df: pd.DataFrame, train_df: pd.DataFrame,
                           target_col: str) -> dict:
        """
        验证生成的测试数据质量

        Returns:
            {'valid': bool, 'issues': [...], 'warnings': [...]}
        """
        issues = []
        warnings = []

        # 1. 行数检查
        if len(test_df) < 20:
            issues.append(f"test set too small: {len(test_df)} rows (need >= 20)")

        # 2. 目标列存在
        if target_col not in test_df.columns:
            issues.append(f"target column '{target_col}' missing in test set")

        # 3. 特征列匹配
        train_cols = set(train_df.columns) - {target_col}
        test_cols = set(test_df.columns) - {target_col}
        missing = train_cols - test_cols
        extra = test_cols - train_cols
        if missing:
            issues.append(f"missing feature columns: {missing}")
        if extra:
            warnings.append(f"extra columns in test set: {extra}")

        # 4. NaN检查
        nan_count = test_df.isnull().sum().sum()
        if nan_count > 0:
            warnings.append(f"test set contains {nan_count} NaN values")

        # 5. 目标列值域检查
        if target_col in test_df.columns and target_col in train_df.columns:
            train_vals = set(train_df[target_col].dropna().unique())
            test_vals = set(test_df[target_col].dropna().unique())
            unseen = test_vals - train_vals
            if unseen:
                warnings.append(f"test set contains {len(unseen)} unseen target values: {list(unseen)[:5]}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
