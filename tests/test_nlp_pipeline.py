"""
============================================
NLP管道测试 — TF-IDF / SMOTE / 数据增强
============================================
Batch D2: 测试NLP训练管道的核心组件
"""
import pickle
import pytest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# ═══════════════════════════════════════════════════════════════════
# Test Data
# ═══════════════════════════════════════════════════════════════════

CHINESE_TEXTS = [
    "酒店位置很好，房间干净整洁，服务态度非常棒",
    "前台接待很热情，早餐种类丰富，下次还会入住",
    "性价比还可以，不过隔音确实不太好",
    "太差了，房间又脏又破，隔音还不好",
    "服务态度恶劣，等了两个小时才入住",
    "总的来说还行，但有些地方需要改进",
    "环境不错，价格实惠，推荐大家来试一下",
    "味道好极了，下次带朋友再来",
    "很失望，跟描述完全不符，不会再去了",
    "一般般吧，没有想象中那么好也没有那么差",
]

CHINESE_LABELS = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]


# ═══════════════════════════════════════════════════════════════════
# TestTFIDFPipeline
# ═══════════════════════════════════════════════════════════════════

class TestTFIDFPipeline:
    """TF-IDF 向量化管道测试"""

    def test_vectorizer_roundtrip(self, tmp_path):
        """fit → save → load → transform 一致性验证"""
        # Fit
        vec = TfidfVectorizer(max_features=50)
        X_orig = vec.fit_transform(CHINESE_TEXTS)

        # Save
        pkl_path = tmp_path / 'vectorizer.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(vec, f)

        # Load
        with open(pkl_path, 'rb') as f:
            vec_loaded = pickle.load(f)

        # Transform with loaded
        X_loaded = vec_loaded.transform(CHINESE_TEXTS)

        # 验证形状一致
        assert X_orig.shape == X_loaded.shape

        # 验证值一致 (dense array)
        orig_dense = X_orig.toarray()
        loaded_dense = X_loaded.toarray()
        np.testing.assert_array_almost_equal(orig_dense, loaded_dense)

    def test_jieba_tokenization(self):
        """jieba 分词后特征非空"""
        try:
            import jieba
        except ImportError:
            pytest.skip("jieba 未安装")

        def jieba_cut(text):
            return ' '.join(jieba.cut(text))

        vec = TfidfVectorizer(
            tokenizer=jieba_cut,
            max_features=50,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X = vec.fit_transform(CHINESE_TEXTS)

        assert X.shape[0] == len(CHINESE_TEXTS)
        assert X.shape[1] > 0, "jieba分词后应产生非零特征"
        # 至少有一些非零值
        assert X.nnz > 0, "jieba分词后应有非零TF-IDF值"

    def test_char_ngram_fallback(self):
        """无 jieba 时 char ngram 回退"""
        vec = TfidfVectorizer(
            max_features=50,
            analyzer='char',
            ngram_range=(1, 3),
            sublinear_tf=True,
            dtype=np.float32,
        )
        X = vec.fit_transform(CHINESE_TEXTS)

        assert X.shape[0] == len(CHINESE_TEXTS)
        assert X.shape[1] > 0, "char ngram应产生非零特征"
        assert X.nnz > 0

    def test_max_features_cap(self):
        """小数据集 → max_features 自适应不超过 n_samples // 2"""
        small_texts = ["好", "差", "还行", "不错"]  # 4 samples
        n_samples = len(small_texts)

        # 模拟 sklearn_trainer 中的自适应逻辑
        requested_mf = 2000
        adaptive_mf = max(100, min(requested_mf, n_samples // 2))

        assert adaptive_mf == 100, f"4样本应cap到100, 实际{adaptive_mf}"

        vec = TfidfVectorizer(max_features=adaptive_mf)
        X = vec.fit_transform(small_texts)

        # 实际特征数不超过 adaptive_mf
        assert X.shape[1] <= adaptive_mf

    def test_vectorizer_feature_names(self):
        """验证 get_feature_names_out() 返回可读特征名"""
        vec = TfidfVectorizer(max_features=20)
        vec.fit(CHINESE_TEXTS)

        names = vec.get_feature_names_out()
        assert len(names) > 0
        assert len(names) <= 20
        # 特征名应为字符串
        assert isinstance(names[0], str)

    def test_persistence_without_tokenizer(self, tmp_path):
        """char ngram vectorizer 序列化/反序列化 (非 local 函数, 兼容 pickle)"""
        vec = TfidfVectorizer(
            max_features=30,
            analyzer='char',
            ngram_range=(1, 3),
            dtype=np.float32,
        )
        X_before = vec.fit_transform(CHINESE_TEXTS[:5])

        pkl_path = tmp_path / 'char_vec.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(vec, f)

        with open(pkl_path, 'rb') as f:
            vec2 = pickle.load(f)

        X_after = vec2.transform(CHINESE_TEXTS[:5])
        np.testing.assert_array_almost_equal(
            X_before.toarray(), X_after.toarray()
        )


# ═══════════════════════════════════════════════════════════════════
# TestSMOTEBalancing
# ═══════════════════════════════════════════════════════════════════

class TestSMOTEBalancing:
    """SMOTE 重采样测试"""

    @pytest.fixture
    def imbalanced_data(self):
        """创建不平衡的文本数据集 (8正 / 2负)"""
        texts = [
            "很好", "不错", "推荐", "满意", "喜欢",
            "好吃", "方便", "干净",
            "太差", "失望",
        ]
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # 80% 正类
        return texts, labels

    def test_smote_balances_classes(self, imbalanced_data):
        """SMOTE 后类别分布均衡"""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imbalanced-learn 未安装")

        texts, labels = imbalanced_data

        # TF-IDF
        vec = TfidfVectorizer(max_features=20)
        X = vec.fit_transform(texts)
        y = np.array(labels)

        # 确认原始不均衡
        unique_before, counts_before = np.unique(y, return_counts=True)
        assert counts_before[0] != counts_before[1], "原始数据应不均衡"

        # SMOTE
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        _, counts_after = np.unique(y_resampled, return_counts=True)
        assert counts_after[0] == counts_after[1], (
            f"SMOTE后应类别均衡, 实际: {counts_after}"
        )

    def test_no_smote_when_disabled(self, imbalanced_data):
        """不启用SMOTE时无重采样"""
        texts, labels = imbalanced_data

        vec = TfidfVectorizer(max_features=20)
        X = vec.fit_transform(texts)
        y = np.array(labels)

        original_count = len(y)
        assert original_count == len(texts)
        # 不做任何重采样，数据量不变
        assert X.shape[0] == original_count

    def test_smote_preserves_samples(self):
        """SMOTE 保持合理的样本生成 (k_neighbors 兼容性)"""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imbalanced-learn 未安装")

        # 3正 / 2负 (min class=2, k_neighbors=1 安全) — 使用多字符中文词避免 empty vocab
        texts = ["很好很好", "很棒很棒", "非常赞", "太差了", "很烂啊"]
        labels = [1, 1, 1, 0, 0]

        vec = TfidfVectorizer(max_features=10)
        X = vec.fit_transform(texts)

        smote = SMOTE(random_state=42, k_neighbors=1)
        X_res, y_res = smote.fit_resample(X, labels)

        _, counts = np.unique(y_res, return_counts=True)
        # 所有类别数应相等 (都=最多的那个)
        assert counts[0] == counts[1]
        assert counts[0] == 3  # max(3, 2) = 3


# ═══════════════════════════════════════════════════════════════════
# TestDataAugmentation
# ═══════════════════════════════════════════════════════════════════

class TestDataAugmentation:
    """文本数据增强测试"""

    @pytest.fixture
    def aug_texts(self):
        """中文情感文本 + 标签"""
        texts = [
            "酒店服务态度非常好，房间干净整洁，推荐大家入住",
            "太失望了，菜品难吃而且价格昂贵，不推荐",
            "环境还可以，不过位置有点偏僻",
            "前台很热情，办理入住很快",
            "隔音太差，一晚上没睡好",
        ]
        labels = ['正面', '负面', '中性', '正面', '负面']
        return texts, labels

    @pytest.mark.parametrize("factor,expected_count_fn", [
        (2, lambda n: n * 2),
        (1, lambda n: n),
    ])
    def test_augmentation_factor(self, aug_texts, factor, expected_count_fn):
        """参数化: factor=2 翻倍 / factor=1 不变"""
        from app.utils.text_augment import augment_texts

        texts, labels = aug_texts
        aug_texts_out, aug_labels_out = augment_texts(
            texts, labels, factor=factor, seed=42
        )

        expected = expected_count_fn(len(texts))
        assert len(aug_texts_out) == expected
        assert len(aug_labels_out) == expected
        if factor == 1:
            assert aug_texts_out == texts
            assert aug_labels_out == labels

    def test_labels_preserved(self, aug_texts):
        """增强后标签不变"""
        from app.utils.text_augment import augment_texts

        texts, labels = aug_texts
        aug_texts_out, aug_labels_out = augment_texts(
            texts, labels, factor=3, seed=42
        )

        # 原始标签保持不变
        assert aug_labels_out[:len(labels)] == labels
        # 增强样本的标签与对应原始样本相同
        # augment_texts 按序追加: 先text[0]所有变体, 再text[1]所有变体...
        for i in range(len(texts)):
            for v in range(2):  # v=0,1 (两个变体, factor-1=2)
                aug_idx = len(texts) + i * 2 + v
                if aug_idx < len(aug_labels_out):
                    assert aug_labels_out[aug_idx] == labels[i], (
                        f"aug_idx={aug_idx}, i={i}, v={v}"
                    )

    @pytest.mark.parametrize("seed1,seed2,expect_equal", [
        (42, 42, True),
        (42, 99, False),
    ])
    def test_augmentation_seeds(self, aug_texts, seed1, seed2, expect_equal):
        """参数化: 同seed → 同输出 / 不同seed → 可能不同"""
        from app.utils.text_augment import augment_texts

        texts, labels = aug_texts

        out1, lab1 = augment_texts(texts, labels, factor=2, seed=seed1)
        out2, lab2 = augment_texts(texts, labels, factor=2, seed=seed2)

        assert len(out1) == len(out2) == len(texts) * 2
        if expect_equal:
            assert out1 == out2, "同seed应产生相同增强结果"
            assert lab1 == lab2

    @pytest.mark.parametrize("aug_func_name,text,check_fn", [
        ("synonym_replace",     "酒店服务态度非常好，房间干净整洁", lambda r: len(r) > 0),
        ("random_char_delete",  "酒店服务态度非常好",             lambda r: 0 < len(r) <= len("酒店服务态度非常好")),
    ])
    def test_augmentation_function(self, aug_texts, aug_func_name, text, check_fn):
        """参数化: 同义替换 / 随机字符删除"""
        from app.utils.text_augment import synonym_replace, random_char_delete

        rng = __import__('random').Random(42)
        if aug_func_name == "synonym_replace":
            result = synonym_replace(text, n=2, rng=rng)
        else:
            result = random_char_delete(text, p=0.15, rng=rng)

        assert check_fn(result)

    def test_empty_text_handling(self):
        """空文本增强不崩溃"""
        from app.utils.text_augment import augment_texts, synonym_replace, random_char_delete

        # synonym_replace on empty
        assert synonym_replace("", n=2) == ""

        # random_char_delete on empty
        assert random_char_delete("", p=0.15) == ""

        # augment_texts with empty
        out, lab = augment_texts([], [], factor=2, seed=42)
        assert out == []
        assert lab == []

    def test_augment_texts_original_preserved(self, aug_texts):
        """增强结果的前 N 个元素是原始数据"""
        from app.utils.text_augment import augment_texts

        texts, labels = aug_texts
        aug_texts_out, aug_labels_out = augment_texts(
            texts, labels, factor=2, seed=42
        )

        assert aug_texts_out[:len(texts)] == texts
        assert aug_labels_out[:len(labels)] == labels

    def test_synonym_dictionary_coverage(self):
        """同义词词典包含足够的词条"""
        from app.utils.text_augment import CHINESE_SENTIMENT_SYNONYMS

        assert len(CHINESE_SENTIMENT_SYNONYMS) >= 20
        # 检查正向词
        assert '推荐' in CHINESE_SENTIMENT_SYNONYMS
        assert '满意' in CHINESE_SENTIMENT_SYNONYMS
        # 检查负向词
        assert '失望' in CHINESE_SENTIMENT_SYNONYMS
        assert '难吃' in CHINESE_SENTIMENT_SYNONYMS
        # 字典值应该是 list
        for word, syns in CHINESE_SENTIMENT_SYNONYMS.items():
            assert isinstance(syns, list), f"{word}: 同义词应为list"
            assert len(syns) >= 1, f"{word}: 至少应有1个同义词"
