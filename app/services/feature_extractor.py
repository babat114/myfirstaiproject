"""
============================================
通用特征提取器
将原始输入 (文本/图像) 转换为模型可用的数值特征向量
用于模型测试页面的多类型输入支持
============================================
"""
import os
import io
import logging
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """通用特征提取器 — 文本/图像 → 数值特征向量

    两种提取模式:
        1. extract_text_features()  — 文本 → TF-IDF 特征向量
        2. extract_image_features() — 图像 → 预训练CNN特征向量

    均为 best-effort 近似转换 (模型训练时使用合成数据, 非真实文本/图像)
    """

    # 缓存: 避免重复加载大型模型
    _tfidf_vectorizer = None
    _cnn_model = None
    _cnn_preprocess = None

    # ============ 文本特征提取 ============

    @staticmethod
    def extract_text_features(
        text: str, n_features: int, language: str = 'zh'
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        将文本转换为 TF-IDF 特征向量, 截断/填充至目标维度

        Args:
            text: 输入文本 (中文/英文)
            n_features: 目标特征维度 (模型期望的输入维度)
            language: 文本语言 ('zh' / 'en'), 影响分词策略

        Returns:
            (features_array, error_message)
            成功: (np.ndarray shape=(1, n_features), None)
            失败: (None, 错误描述)
        """
        if not text or not text.strip():
            return None, '请输入文本内容。'

        text = text.strip()

        # 检查文本长度
        if len(text) < 2:
            return None, '文本太短 (少于2个字符)，请输入更长的内容。'

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            # 中文分词 (如果安装了 jieba)
            if language == 'zh':
                try:
                    import jieba
                    text = ' '.join(jieba.cut(text))
                except ImportError:
                    # 无 jieba 则按字符级处理 (单字 gram)
                    text = ' '.join(list(text))

            # 配置 TF-IDF (字符级 ngram 1-2, 适应不同文本长度)
            vectorizer = TfidfVectorizer(
                analyzer='char' if language == 'zh' else 'word',
                ngram_range=(1, 2),
                max_features=n_features,
                sublinear_tf=True,  # 1 + log(tf) 平滑
                dtype=np.float32,
            )

            # Fit on this text + build vocabulary
            # 使用通用语料增强词汇覆盖 (少量中文常见字)
            if language == 'zh':
                dummy_corpus = [
                    ' '.join(list('这是一个测试文本用于机器学习模型预测分析')),
                    ' '.join(list('自然语言处理是人工智能的重要分支')),
                    text,
                ]
            else:
                dummy_corpus = [
                    'this is a test document for machine learning prediction',
                    'natural language processing is an important branch of ai',
                    text,
                ]
            vectorizer.fit(dummy_corpus)
            features = vectorizer.transform([text]).toarray().astype(np.float32)

            # 对齐特征维度
            features = FeatureExtractor._pad_or_truncate(features, n_features)

            logger.info(
                f'文本特征提取完成: {len(text)}字符 -> {n_features}维特征, '
                f'词汇量={len(vectorizer.vocabulary_)}'
            )
            return features, None

        except ImportError as e:
            return None, f'缺少依赖: {e}。请安装 scikit-learn。'
        except Exception as e:
            logger.error(f'文本特征提取失败: {e}', exc_info=True)
            return None, f'文本转换失败: {str(e)}'

    # ============ 图像特征提取 ============

    @staticmethod
    def extract_image_features(
        image_data: bytes, n_features: int
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        从图像字节数据中提取特征向量

        优先使用 PyTorch 预训练 ResNet-18 (512维 → 截断/填充至 n_features)
        PyTorch 不可用时回退到经典 CV 特征 (HOG + 颜色直方图)

        Args:
            image_data: 图像文件的字节内容 (PNG/JPG/WebP等)
            n_features: 目标特征维度

        Returns:
            (features_array, error_message)
        """
        if not image_data:
            return None, '请上传图像文件。'

        # 尝试 PyTorch 路径
        features, error = FeatureExtractor._extract_with_cnn(image_data, n_features)
        if features is not None:
            return features, None

        # 回退到经典 CV 特征
        logger.warning(f'CNN特征提取失败 ({error})，回退到经典CV特征')
        return FeatureExtractor._extract_with_classical_cv(image_data, n_features)

    @staticmethod
    def _extract_with_cnn(
        image_data: bytes, n_features: int
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """使用预训练 ResNet-18 提取特征 (512维)"""
        try:
            import torch
            from torchvision import models, transforms
            from PIL import Image

            # 懒加载模型 (只加载一次)
            if FeatureExtractor._cnn_model is None:
                logger.info('加载预训练 ResNet-18 特征提取器...')
                model = models.resnet18(weights='DEFAULT')
                # 去掉最后的全连接层，保留全局平均池化输出 (512维)
                FeatureExtractor._cnn_model = torch.nn.Sequential(
                    *list(model.children())[:-1]
                )
                FeatureExtractor._cnn_model.eval()
                FeatureExtractor._cnn_preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])

            # 打开图像
            img = Image.open(io.BytesIO(image_data)).convert('RGB')

            # 预处理
            img_tensor = FeatureExtractor._cnn_preprocess(img).unsqueeze(0)

            # 提取特征
            with torch.no_grad():
                features = FeatureExtractor._cnn_model(img_tensor)
                features = features.squeeze().numpy().reshape(1, -1).astype(np.float32)

            # 对齐维度
            features = FeatureExtractor._pad_or_truncate(features, n_features)

            logger.info(
                f'CNN 特征提取完成: {img.size} -> {n_features}维特征'
            )
            return features, None

        except ImportError as e:
            return None, f'缺少PyTorch/torchvision: {e}'
        except Exception as e:
            logger.error(f'CNN特征提取失败: {e}', exc_info=True)
            return None, f'CNN特征提取失败: {str(e)}'

    @staticmethod
    def _extract_with_classical_cv(
        image_data: bytes, n_features: int
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """回退方案: 使用 HOG + 颜色直方图提取经典CV特征"""
        try:
            from PIL import Image
            from skimage.feature import hog
            from skimage import exposure

            img = Image.open(io.BytesIO(image_data)).convert('L')  # 灰度

            # 缩放到固定大小 (平衡细节和速度)
            img.thumbnail((128, 128), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0

            # HOG 特征 (纹理/形状)
            try:
                hog_features = hog(
                    img_array,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    feature_vector=True,
                )
            except Exception:
                hog_features = np.array([])

            # 颜色直方图 (亮度分布)
            hist, _ = exposure.histogram(img_array, nbins=64)
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist = hist / hist.sum()

            # 简单统计特征
            stats = np.array([
                np.mean(img_array),
                np.std(img_array),
                np.min(img_array),
                np.max(img_array),
                np.median(img_array),
            ], dtype=np.float32)

            # 拼接所有特征
            all_features = np.concatenate([hog_features, hist, stats])
            features = all_features.reshape(1, -1).astype(np.float32)

            # 对齐维度
            features = FeatureExtractor._pad_or_truncate(features, n_features)

            logger.info(
                f'经典CV特征提取完成: HOG={len(hog_features)}维 + 直方图=64维 + 统计=5维 -> {n_features}维'
            )
            return features, None

        except ImportError as e:
            return None, f'缺少图像处理依赖: {e}。请安装 scikit-image, Pillow。'
        except Exception as e:
            logger.error(f'经典CV特征提取失败: {e}', exc_info=True)
            return None, f'图像处理失败: {str(e)}'

    # ============ 工具方法 ============

    @staticmethod
    def _pad_or_truncate(features: np.ndarray, target_dim: int) -> np.ndarray:
        """
        将特征向量截断或零填充至目标维度

        Args:
            features: shape (1, d) 的特征数组
            target_dim: 目标维度

        Returns:
            shape (1, target_dim) 的特征数组
        """
        current_dim = features.shape[1]
        if current_dim == target_dim:
            return features
        elif current_dim > target_dim:
            # 截断 (保留最显著的特征)
            return features[:, :target_dim]
        else:
            # 零填充
            padded = np.zeros((1, target_dim), dtype=np.float32)
            padded[:, :current_dim] = features
            return padded

    @staticmethod
    def load_image_thumbnail(image_data: bytes, max_size: Tuple[int, int] = (224, 224)) -> Optional[bytes]:
        """
        生成缩略图用于前端预览 (返回 PNG bytes)

        Args:
            image_data: 原始图像字节
            max_size: 缩略图最大尺寸
        """
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail(max_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return buf.getvalue()
        except Exception:
            return None

    # ============ 中文情感分析 (NLP 模型 fallback) ============

    # 中文情感词典 (正面/负面关键词)
    _POSITIVE_WORDS: set = {
        '好', '棒', '优秀', '出色', '完美', '喜欢', '爱', '赞', '精彩', '推荐',
        '满意', '不错', '给力', '厉害', '牛', '强', '快乐', '开心', '高兴', '幸福',
        '好看', '好听', '好吃', '好玩', '舒服', '方便', '实用', '简单', '轻松',
        '便宜', '划算', '实惠', '值', '超值', '惊艳', '惊喜', '感动', '温暖',
        '专业', '认真', '负责', '细心', '耐心', '热情', '友好', '靠谱', '值得',
        '放心', '安全', '稳定', '流畅', '清晰', '漂亮', '精致', '高级', '大气',
        '不错哦', '太棒了', '非常好', '很不错', '太好了', '真不错', '好评',
        '物超所值', '性价比高', '良心', '顶级', '一流', '无敌', '绝了', '神',
        '优雅', '舒适', '便捷', '高效', '智能', '创新', '领先', '突破',
    }
    _NEGATIVE_WORDS: set = {
        '差', '烂', '糟糕', '失望', '讨厌', '恶心', '垃圾', '坑', '骗', '假',
        '不行', '不好', '难用', '复杂', '麻烦', '慢', '卡', '丑', '难听', '难吃',
        '贵', '不值', '坑爹', '生气', '愤怒', '伤心', '难过', '无聊', '烦',
        '垃圾货', '太差了', '很差', '非常差', '差评', '不好用', '太难了',
        '质量差', '服务差', '态度差', '不专业', '不负责', '敷衍', '忽悠',
        '不安全', '不稳定', '故障', 'bug', '崩溃', '闪退', '卡顿', '延迟',
        '粗糙', '劣质', '山寨', '盗版', '假冒', '虚假', '误导', '欺骗',
        '坑人', '黑心', '无良', '差劲', '低劣', '不堪', '恶劣',
    }
    _NEGATION_WORDS: set = {
        '不', '没', '无', '非', '别', '未', '否', '勿', '莫', '休',
    }

    @staticmethod
    def analyze_sentiment(text: str) -> dict:
        """
        中文情感关键词分析 — NLP 模型无向量化器时的 fallback

        统计正面/负面关键词出现次数, 考虑否定词翻转。
        返回: {label: "正面/负面/中性", confidence: 0-1, positive_count, negative_count}
        """
        text = text.strip()
        if not text:
            return {'label': '中性', 'confidence': 0.0,
                    'positive_count': 0, 'negative_count': 0}

        # 尝试用 jieba 分词, 不可用则按字符滑动窗口匹配
        words = []
        try:
            import jieba
            words = list(jieba.cut(text))
        except ImportError:
            # 按2-4字滑动窗口 + 单字
            words = list(text)
            for win in [2, 3, 4]:
                for i in range(len(text) - win + 1):
                    words.append(text[i:i + win])

        pos_count = 0
        neg_count = 0
        words_list = list(words)

        for i, w in enumerate(words_list):
            w_clean = w.strip()
            if not w_clean:
                continue

            if w_clean in FeatureExtractor._POSITIVE_WORDS:
                # 检查前面是否有否定词
                negated = False
                for j in range(max(0, i - 2), i):
                    if words_list[j].strip() in FeatureExtractor._NEGATION_WORDS:
                        negated = True
                        break
                if negated:
                    neg_count += 1
                else:
                    pos_count += 1

            elif w_clean in FeatureExtractor._NEGATIVE_WORDS:
                negated = False
                for j in range(max(0, i - 2), i):
                    if words_list[j].strip() in FeatureExtractor._NEGATION_WORDS:
                        negated = True
                        break
                if negated:
                    pos_count += 1  # 否定负面 → 正面
                else:
                    neg_count += 1

        total = pos_count + neg_count
        if total == 0:
            return {'label': '中性', 'confidence': 0.0,
                    'positive_count': 0, 'negative_count': 0}

        if pos_count > neg_count:
            label = '正面'
            confidence = round(pos_count / total, 3)
        elif neg_count > pos_count:
            label = '负面'
            confidence = round(neg_count / total, 3)
        else:
            label = '中性'
            confidence = 0.5

        return {
            'label': label,
            'confidence': confidence,
            'positive_count': pos_count,
            'negative_count': neg_count,
        }
