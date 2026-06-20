"""
Chinese text data augmentation for NLP sentiment analysis.
No external API dependencies — pure dictionary-based synonym replacement
and character-level augmentation.

Techniques:
  1. Synonym replacement: small curated Chinese sentiment synonym dictionary
  2. Random character deletion: delete 10-20% of characters randomly
"""
import random

# ===================================================================
# Curated Chinese Sentiment Synonym Dictionary (~50 word pairs)
# Grouped by sentiment polarity to avoid flipping sentiment
# ===================================================================
CHINESE_SENTIMENT_SYNONYMS = {
    # --- Positive words ---
    '推荐': ['值得', '建议', '力荐'],
    '精彩': ['出色', '优秀', '惊艳'],
    '满意': ['满足', '称心', '如意'],
    '实惠': ['便宜', '划算', '合算'],
    '干净': ['整洁', '清爽', '卫生'],
    '热情': ['热心', '周到', '体贴'],
    '不错': ['挺好', '可以', '还行'],
    '好吃': ['美味', '可口', '好吃'],
    '好看': ['精美', '漂亮', '美观'],
    '舒服': ['舒适', '惬意', '舒坦'],
    '安静': ['清净', '幽静', '宁静'],
    '方便': ['便利', '便捷', '省事'],
    '细心': ['细致', '周到', '用心'],
    '正宗': ['地道', '纯正', '原味'],
    '值得': ['超值', '划算', '值得'],
    # --- Negative words ---
    '失望': ['沮丧', '灰心', '遗憾'],
    '脏乱': ['邋遢', '污浊', '杂乱'],
    '吵闹': ['嘈杂', '喧哗', '烦人'],
    '冷漠': ['冷淡', '漠视', '无情'],
    '粗糙': ['毛糙', '粗劣', '简陋'],
    '昂贵': ['高价', '不值', '宰人'],
    '难吃': ['恶心', '反胃', '倒胃口'],
    '拥挤': ['拥堵', '人满为患', '水泄不通'],
    '混乱': ['杂乱', '无序', '一塌糊涂'],
    '耽误': ['延误', '拖延', '浪费时间'],
    # --- Neutral-to-negative words ---
    '一般': ['普通', '平平', '中规中矩'],
    '凑合': ['将就', '勉强', '马马虎虎'],
}


def _find_synonym_candidates(text: str, max_replace: int = 2) -> list:
    """Find words in text that have synonyms in the dictionary.
    Returns list of (start, end, synonyms_list) tuples.
    """
    candidates = []
    for word, synonyms in CHINESE_SENTIMENT_SYNONYMS.items():
        idx = 0
        while True:
            idx = text.find(word, idx)
            if idx == -1:
                break
            candidates.append((idx, idx + len(word), synonyms))
            idx += 1

    if len(candidates) > max_replace:
        candidates = random.sample(candidates, max_replace)
    return candidates


def synonym_replace(text: str, n: int = 2, rng: random.Random = None) -> str:
    """Replace up to n words in text with synonyms from the dictionary.

    Args:
        text: Chinese text string
        n: Maximum number of words to replace
        rng: Optional Random instance for reproducibility

    Returns:
        Augmented text with some words replaced by synonyms
    """
    if rng is None:
        rng = random
    candidates = _find_synonym_candidates(text, max_replace=n)
    if not candidates:
        return text

    # Apply replacements from end to start to preserve indices
    candidates.sort(key=lambda x: x[0], reverse=True)
    result = text
    for start, end, synonyms in candidates:
        replacement = rng.choice(synonyms) if hasattr(rng, 'choice') else random.choice(synonyms)
        result = result[:start] + replacement + result[end:]

    return result


def random_char_delete(text: str, p: float = 0.15, rng: random.Random = None) -> str:
    """Randomly delete p% of characters from text.

    Simulates typos and informal writing, increasing robustness
    to character-level variation in Chinese text.

    Args:
        text: Chinese text string
        p: Deletion probability per character (0.10 to 0.20 recommended)
        rng: Optional Random instance for reproducibility

    Returns:
        Text with some characters randomly removed
    """
    if rng is None:
        rng = random
    if not text:
        return text

    chars = list(text)
    if hasattr(rng, 'random'):
        kept = [c for c in chars if rng.random() > p]
    else:
        kept = [c for c in chars if random.random() > p]

    if not kept:
        # Keep at least 30% of original length
        n_keep = max(1, int(len(chars) * 0.3))
        if hasattr(rng, 'sample'):
            kept = rng.sample(chars, n_keep)
        else:
            kept = random.sample(chars, n_keep)

    return ''.join(kept)


def augment_texts(texts, labels, factor=2, methods=None, seed=42):
    """Augment a list of texts by the given factor.

    For factor=2: each original text generates 1 augmented variant (doubles).
    For factor=3: each original text generates 2 augmented variants (triples).

    Augmentation methods are applied in alternation:
      - 'synonym': synonym replacement (2 words)
      - 'delete': random character deletion (10-20%)

    Original texts + augmented variants are returned together.

    Args:
        texts: List of Chinese text strings
        labels: List of corresponding labels
        factor: Augmentation factor (1 = no augmentation, 2 = double, etc.)
        methods: List of methods (default: ['synonym', 'delete'])
        seed: Random seed for reproducibility

    Returns:
        (augmented_texts, augmented_labels) — both lists
    """
    if factor <= 1:
        return list(texts), list(labels)

    if methods is None:
        methods = ['synonym', 'delete']

    rng = random.Random(seed)

    aug_texts = list(texts)
    aug_labels = list(labels)

    for i, (text, label) in enumerate(zip(texts, labels)):
        for v in range(factor - 1):
            method = methods[(i + v) % len(methods)]
            local_rng = random.Random(seed + i * 100 + v)

            if method == 'synonym':
                augmented = synonym_replace(text, n=2, rng=local_rng)
            elif method == 'delete':
                p = local_rng.uniform(0.10, 0.20)
                augmented = random_char_delete(text, p=p, rng=local_rng)
            else:
                augmented = text

            aug_texts.append(augmented)
            aug_labels.append(label)

    return aug_texts, aug_labels
