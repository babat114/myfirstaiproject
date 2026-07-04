"""
共享 NLP 预处理模块 — sklearn 和 PyTorch 训练器共用

提供:
  - jieba 分词器
  - NLP 文本列检测
  - TfidfVectorizer 配置生成
  - train/val/test 3-way TF-IDF 变换 (仅在训练集上拟合)
  - 类别标签保存

设计原则: 与训练器框架无关, 接受 DataFrame/ndarray, 返回处理后的数据
"""
import logging
import numpy as np
import pandas as pd

_nlp_logger = logging.getLogger(__name__)


# ============ Tokenizer (必须可 pickle) ============

def jieba_tokenize(text: str):
    """jieba 中文分词 — 模块级函数 (必须可 pickle)"""
    import jieba
    return list(jieba.cut(text))


def combined_tokenize(text: str):
    """合并分词: jieba 中文分词 + 中文字符 unigram.

    确保短文本 (如"垃圾" "勉强" "烂") 即使不在 jieba 词典中,
    也能通过字符级特征命中 TF-IDF 词汇表, 消除 nnz=0 问题。

    返回 jieba 词语 + 单个中文字符的列表。
    模块级函数 (必须可 pickle 以支持 TfidfVectorizer 序列化)。
    """
    import jieba
    import re
    words = list(jieba.cut(text))
    # 提取中文字符 unigram (U+4E00-U+9FFF 基本汉字 + U+3400-U+4DBF 扩展A)
    chars = re.findall(r'[㐀-鿿]', text)
    return words + chars


# ============ NLP 文本列检测 ============

TEXT_COLUMN_CANDIDATES = ['text', 'review', 'comment', 'content', 'sentence']


def detect_nlp_text_column(df: pd.DataFrame, dataset_category: str = None):
    """检测 DataFrame 中的 NLP 文本列。

    Args:
        df: 数据 DataFrame
        dataset_category: 数据集的 category 字段 (如 'nlp')

    Returns:
        text_col_name (str|None): 检测到的文本列名, 或 None
    """
    if dataset_category != 'nlp':
        return None

    # 如果第一列已经是 tfidf_* 说明数据已被预处理
    if df.columns[0].startswith('tfidf_'):
        _nlp_logger.info('[NLP] Features already TF-IDF, skipping')
        return None

    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in df.columns:
            _nlp_logger.info('[NLP] Text column found: %s', candidate)
            return candidate

    _nlp_logger.info('[NLP] No text column found in %s', list(df.columns)[:8])
    return None


# ============ Vectorizer 配置 ============

def create_vectorizer_config(nlp_max_features=2000, nlp_min_df=2, nlp_max_df=0.9):
    """创建 TfidfVectorizer 配置字典。

    优先使用 jieba 分词, 回退到 char ngram。

    Args:
        nlp_max_features: 最大特征数
        nlp_min_df: 最小文档频率
        nlp_max_df: 最大文档频率
    """
    try:
        import jieba  # noqa: F401
        _nlp_logger.info('[NLP] Using combined jieba+char tokenizer')
        return {
            'tokenizer': combined_tokenize,
            'max_features': nlp_max_features,
            'min_df': nlp_min_df,
            'max_df': nlp_max_df,
            'sublinear_tf': True,
            'dtype': np.float32,
        }
    except ImportError:
        _nlp_logger.info('[NLP] jieba not available, falling back to char ngram')
        return {
            'max_features': nlp_max_features,
            'min_df': nlp_min_df,
            'max_df': nlp_max_df,
            'analyzer': 'char',
            'ngram_range': (1, 3),
            'sublinear_tf': True,
            'dtype': np.float32,
        }


# ============ 3-way TF-IDF 变换 ============

def apply_tfidf_3way(
    train_indices: list,
    val_indices: list,
    test_indices: list,
    nlp_texts: list,
    vectorizer_config: dict,
):
    """在训练集上拟合 TfidfVectorizer, 然后变换 train/val/test 三组。

    Args:
        train_indices: 训练样本在原始文本列表中的索引
        val_indices: 验证样本索引
        test_indices: 测试样本索引
        nlp_texts: 完整文本列表 (按原始 DataFrame 顺序)
        vectorizer_config: TfidfVectorizer 参数字典

    Returns:
        (vectorizer, tfidf_train, tfidf_val, tfidf_test, n_features)
        每个 tfidf_* 是 scipy sparse matrix (CSR)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_texts = [nlp_texts[i] for i in train_indices]
    val_texts = [nlp_texts[i] for i in val_indices] if val_indices else []
    test_texts = [nlp_texts[i] for i in test_indices]

    vectorizer = TfidfVectorizer(**vectorizer_config)
    tfidf_train = vectorizer.fit_transform(train_texts)
    tfidf_val = vectorizer.transform(val_texts) if val_texts else None
    tfidf_test = vectorizer.transform(test_texts)

    n_features = tfidf_train.shape[1]
    _nlp_logger.info(
        '[NLP] TF-IDF complete: %d features, train=%d val=%d test=%d',
        n_features, tfidf_train.shape[0],
        tfidf_val.shape[0] if tfidf_val is not None else 0,
        tfidf_test.shape[0],
    )

    return vectorizer, tfidf_train, tfidf_val, tfidf_test, n_features


def tfidf_to_dataframe(tfidf_matrix, indices, prefix='tfidf'):
    """将 TF-IDF sparse matrix 转为 DataFrame (列名 tfidf_0, tfidf_1, ...)

    保持稀疏格式避免 OOM: 使用 from_spmatrix 而非 .toarray() 密集化。
    100k 样本 × 2000 特征的 TF-IDF:
      - .toarray(): ~1.6GB 内存
      - from_spmatrix: ~几十MB (视稀疏度而定)
    """
    if tfidf_matrix is None:
        return None
    n_cols = tfidf_matrix.shape[1]
    cols = [f'{prefix}_{i}' for i in range(n_cols)]
    result = pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, index=indices, columns=cols)
    return result


# ============ 类别标签 ============

def extract_class_labels(y: pd.Series) -> list:
    """从目标列提取人类可读的类别标签 (排序)。"""
    try:
        unique_labels = y.dropna().unique()
        return [str(l) for l in sorted(unique_labels, key=str)]
    except Exception:
        return []
