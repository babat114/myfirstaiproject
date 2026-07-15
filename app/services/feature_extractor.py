"""
============================================
通用特征提取器
将原始输入 (文本/图像) 转换为模型可用的数值特征向量
用于模型测试页面的多类型输入支持

v2.1 优化:
  - 多尺度 CNN 特征提取 (3 scales → 平均池化, 抗缩放鲁棒)
  - L2 归一化特征 (余弦相似度匹配)
  - 图像相似度对比 compare_images()
  - 经典 CV 增强: LBP 纹理 + ORB 关键点统计
============================================
"""
import io
import logging

import numpy as np

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
    ) -> tuple[np.ndarray | None, str | None]:
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
        image_data: bytes, n_features: int,
        multi_scale: bool = True
    ) -> tuple[np.ndarray | None, str | None]:
        """
        从图像字节数据中提取特征向量

        优先使用 PyTorch 预训练 ResNet-18 (512维 → 截断/填充至 n_features)
        PyTorch 不可用时回退到经典 CV 特征 (HOG + LBP + 颜色直方图)

        Args:
            image_data: 图像文件的字节内容 (PNG/JPG/WebP等)
            n_features: 目标特征维度
            multi_scale: 是否使用多尺度提取 (默认True, 3尺度平均对缩放更鲁棒)

        Returns:
            (features_array, error_message)
        """
        if not image_data:
            return None, '请上传图像文件。'

        # 尝试 PyTorch 路径
        features, error = FeatureExtractor._extract_with_cnn(
            image_data, n_features, multi_scale=multi_scale
        )
        if features is not None:
            return features, None

        # 回退到经典 CV 特征
        logger.warning(f'CNN特征提取失败 ({error})，回退到经典CV特征')
        return FeatureExtractor._extract_with_classical_cv(image_data, n_features)

    @staticmethod
    def _extract_with_cnn(
        image_data: bytes, n_features: int, multi_scale: bool = True
    ) -> tuple[np.ndarray | None, str | None]:
        """使用预训练 ResNet-18 提取多尺度特征 (3尺度→平均池化→L2归一化)

        多尺度策略:
          1. 基准尺度: resize(256) → center_crop(224) → 512维
          2. 放大尺度: resize(320) → center_crop(224) → 512维 (1.25x, 捕获细节)
          3. 缩小尺度: resize(192) → center_crop(224) → 512维 (0.75x, 全局上下文)
          平均池化 → 512维 → L2归一化 → 对齐至 n_features

        单尺度模式 (multi_scale=False):
          resize(256) → center_crop(224) → 512维 → L2归一化 → 对齐
        """
        try:
            import torch
            from PIL import Image
            from torchvision import models, transforms

            # 懒加载模型 (只加载一次)
            if FeatureExtractor._cnn_model is None:
                logger.info('加载预训练 ResNet-18 特征提取器...')
                model = models.resnet18(weights='DEFAULT')
                FeatureExtractor._cnn_model = torch.nn.Sequential(
                    *list(model.children())[:-1]
                )
                FeatureExtractor._cnn_model.eval()

            # 打开图像
            img = Image.open(io.BytesIO(image_data)).convert('RGB')

            # 基础预处理 (除 resize 外)
            base_transforms = transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            if multi_scale:
                # 三尺度提取
                scale_sizes = [256, 320, 192]  # base, zoom-in, zoom-out
                all_features = []
                for size in scale_sizes:
                    # 保持宽高比缩放到短边≥size, 然后中心裁剪
                    img_resized = img.copy()
                    img_resized.thumbnail((size, size), Image.LANCZOS)
                    # 如果缩略图小于目标, 填充到至少 size×size
                    if img_resized.size[0] < 224 or img_resized.size[1] < 224:
                        new_img = Image.new('RGB', (max(img_resized.size[0], 224),
                                           max(img_resized.size[1], 224)), (0, 0, 0))
                        new_img.paste(img_resized, (
                            (new_img.size[0] - img_resized.size[0]) // 2,
                            (new_img.size[1] - img_resized.size[1]) // 2,
                        ))
                        img_resized = new_img
                    img_tensor = base_transforms(img_resized).unsqueeze(0)
                    with torch.no_grad():
                        feats = FeatureExtractor._cnn_model(img_tensor)
                        feats = feats.squeeze().numpy().reshape(1, -1).astype(np.float32)
                    all_features.append(feats)

                # 平均池化 (抗尺度变化)
                features = np.mean(all_features, axis=0)
            else:
                # 单尺度 (兼容旧行为)
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                img_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    features = FeatureExtractor._cnn_model(img_tensor)
                    features = features.squeeze().numpy().reshape(1, -1).astype(np.float32)

            # L2 归一化 (余弦相似度匹配)
            features = FeatureExtractor._l2_normalize(features)

            # 对齐维度
            features = FeatureExtractor._pad_or_truncate(features, n_features)

            logger.info(
                f'CNN 特征提取完成 (multi_scale={multi_scale}): '
                f'{img.size} -> {n_features}维特征 (L2归一化)'
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
    ) -> tuple[np.ndarray | None, str | None]:
        """回退方案: 使用 HOG + LBP 纹理 + 颜色直方图 + ORB关键点统计

        增强说明 (v2.1):
          - HOG: 梯度方向直方图 (形状/边缘)
          - LBP: 局部二值模式 (纹理, 对光照不敏感)
          - 颜色直方图: 全局亮度分布 (64 bins)
          - ORB 关键点统计: 关键点数量+平均响应 (特征丰富度)
          - 基础统计: mean/std/min/max/median
        """
        try:
            from PIL import Image
            from skimage import exposure
            from skimage.feature import hog, local_binary_pattern

            img = Image.open(io.BytesIO(image_data)).convert('L')  # 灰度

            # 缩放到固定大小 (平衡细节和速度)
            img.thumbnail((128, 128), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_uint8 = (img_array * 255).astype(np.uint8)

            feature_parts = []

            # 1. HOG 特征 (纹理/形状)
            try:
                hog_features = hog(
                    img_array,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    feature_vector=True,
                )
                feature_parts.append(hog_features)
            except Exception:
                pass

            # 2. LBP 纹理特征 (对光照变化鲁棒)
            try:
                radius = 1
                n_points = 8 * radius
                lbp = local_binary_pattern(img_uint8, n_points, radius, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 4),
                                           range=(0, n_points + 3))
                lbp_hist = lbp_hist.astype(np.float32)
                if lbp_hist.sum() > 0:
                    lbp_hist = lbp_hist / lbp_hist.sum()
                feature_parts.append(lbp_hist)
            except Exception:
                pass

            # 3. ORB 关键点统计 (特征丰富度, 匹配质量指标)
            try:
                from skimage.feature import ORB
                detector = ORB(n_keypoints=50, fast_n=9)
                detector.detect_and_extract(img_uint8)
                n_kp = len(detector.keypoints)
                kp_stats = np.array([
                    float(n_kp),
                    float(np.mean(detector.responses)) if n_kp > 0 else 0.0,
                    float(np.std(detector.responses)) if n_kp > 1 else 0.0,
                    float(np.max(detector.responses)) if n_kp > 0 else 0.0,
                ], dtype=np.float32)
                feature_parts.append(kp_stats)
            except Exception:
                # ORB 不可用则不添加
                pass

            # 4. 颜色直方图 (亮度分布)
            hist, _ = exposure.histogram(img_array, nbins=64)
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist = hist / hist.sum()
            feature_parts.append(hist)

            # 5. 简单统计特征
            stats = np.array([
                np.mean(img_array),
                np.std(img_array),
                np.min(img_array),
                np.max(img_array),
                np.median(img_array),
            ], dtype=np.float32)
            feature_parts.append(stats)

            # 拼接所有特征 → L2归一化
            all_features = np.concatenate(feature_parts)
            features = all_features.reshape(1, -1).astype(np.float32)
            features = FeatureExtractor._l2_normalize(features)

            # 对齐维度
            features = FeatureExtractor._pad_or_truncate(features, n_features)

            dims = [len(p) for p in feature_parts]
            logger.info(
                f'经典CV特征提取完成: HOG={dims[0]} + LBP={dims[1] if len(dims)>1 else 0}'
                f' + ORB_kp={dims[2] if len(dims)>2 else 0} + Hist=64 + Stats=5'
                f' = {sum(dims)} -> {n_features}维 (L2归一化)'
            )
            return features, None

        except ImportError as e:
            return None, f'缺少图像处理依赖: {e}。请安装 scikit-image, Pillow。'
        except Exception as e:
            logger.error(f'经典CV特征提取失败: {e}', exc_info=True)
            return None, f'图像处理失败: {str(e)}'

    # ============ 工具方法 ============

    @staticmethod
    def _l2_normalize(features: np.ndarray) -> np.ndarray:
        """L2 归一化特征向量，使 ||v||_2 = 1

        归一化后特征可直接用于余弦相似度计算 (dot product = cosine sim)
        零向量保持为零向量 (避免除零)
        """
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        # 避免除零: 零向量保持不变
        norm = np.where(norm == 0, 1.0, norm)
        return features / norm

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
            # 截断 (保留最显著的特征 — 已按重要性排序)
            return features[:, :target_dim]
        else:
            # 零填充
            padded = np.zeros((1, target_dim), dtype=np.float32)
            padded[:, :current_dim] = features
            return padded

    @staticmethod
    def compare_images(
        image_data_a: bytes,
        image_data_b: bytes,
        n_features: int = 512,
    ) -> dict:
        """比较两张图像的相似度 (基于 CNN 特征余弦相似度)

        用于图像去重、相似图搜索、匹配验证等场景。

        Args:
            image_data_a: 图像 A 的字节内容
            image_data_b: 图像 B 的字节内容
            n_features: 特征维度 (默认512, 与 ResNet-18 输出对齐)

        Returns:
            {
                'similarity': float,      # 余弦相似度 [-1, 1], ≥0.90 高度相似
                'distance': float,        # 余弦距离 [0, 2], 0=完全相同
                'feat_dims': int,         # 实际特征维度
                'method': str,            # 'cnn' 或 'classical_cv'
                'match_level': str,       # 'identical'/'high'/'medium'/'low'/'none'
                'error': str | None,
            }
        """
        if not image_data_a or not image_data_b:
            return {
                'similarity': 0.0, 'distance': 2.0, 'feat_dims': 0,
                'method': 'none', 'match_level': 'none',
                'error': '请提供两张图像的数据。',
            }

        try:
            # 提取两张图的特征
            feats_a, err_a = FeatureExtractor.extract_image_features(
                image_data_a, n_features
            )
            feats_b, err_b = FeatureExtractor.extract_image_features(
                image_data_b, n_features
            )

            method = 'cnn'
            # 如果 CNN 失败则回退到经典 CV
            if err_a and 'CNN' in (err_a or ''):
                feats_a, err_a = FeatureExtractor._extract_with_classical_cv(
                    image_data_a, n_features
                )
                method = 'classical_cv'
            if err_b and 'CNN' in (err_b or ''):
                feats_b, err_b = FeatureExtractor._extract_with_classical_cv(
                    image_data_b, n_features
                )
                method = 'classical_cv'

            if err_a:
                return {
                    'similarity': 0.0, 'distance': 2.0, 'feat_dims': 0,
                    'method': method, 'match_level': 'none',
                    'error': f'图像A处理失败: {err_a}',
                }
            if err_b:
                return {
                    'similarity': 0.0, 'distance': 2.0, 'feat_dims': 0,
                    'method': method, 'match_level': 'none',
                    'error': f'图像B处理失败: {err_b}',
                }

            # 余弦相似度 = dot(A, B) / (||A|| * ||B||)
            # 特征已经 L2 归一化，所以直接 dot product
            sim = float(np.dot(feats_a, feats_b.T)[0, 0])
            # 钳制到 [-1, 1] (浮点精度可能导致微小越界)
            sim = max(-1.0, min(1.0, sim))
            dist = 1.0 - sim  # 余弦距离

            # 匹配等级
            if sim >= 0.99:
                level = 'identical'
            elif sim >= 0.90:
                level = 'high'
            elif sim >= 0.70:
                level = 'medium'
            elif sim >= 0.50:
                level = 'low'
            else:
                level = 'none'

            logger.info(
                f'图像对比完成: similarity={sim:.4f}, level={level}, '
                f'method={method}, dims={feats_a.shape[1]}'
            )
            return {
                'similarity': round(sim, 4),
                'distance': round(dist, 4),
                'feat_dims': int(feats_a.shape[1]),
                'method': method,
                'match_level': level,
                'error': None,
            }

        except Exception as e:
            logger.error(f'图像对比失败: {e}', exc_info=True)
            return {
                'similarity': 0.0, 'distance': 2.0, 'feat_dims': 0,
                'method': 'none', 'match_level': 'none',
                'error': f'对比失败: {str(e)}',
            }

    @staticmethod
    def batch_compare(
        image_data: bytes,
        candidates: list[bytes],
        n_features: int = 512,
        top_k: int = 5,
    ) -> dict:
        """批量对比: 给定一张查询图, 从候选集中找出最相似的 top_k 张

        Args:
            image_data: 查询图像字节
            candidates: 候选图像字节列表
            n_features: 特征维度
            top_k: 返回前 k 个最相似的结果

        Returns:
            {'matches': [{index, similarity, distance, match_level}, ...], 'query_dims': int}
        """
        if not image_data or not candidates:
            return {'matches': [], 'query_dims': 0, 'error': '查询图或候选集为空'}

        # 提取查询图特征
        query_feat, err = FeatureExtractor.extract_image_features(
            image_data, n_features
        )
        if err:
            # 回退到经典 CV
            query_feat, err = FeatureExtractor._extract_with_classical_cv(
                image_data, n_features
            )
        if err:
            return {'matches': [], 'query_dims': 0, 'error': f'查询图处理失败: {err}'}

        results = []
        for idx, cand in enumerate(candidates):
            try:
                cand_feat, _ = FeatureExtractor.extract_image_features(
                    cand, n_features
                )
                if cand_feat is None:
                    continue
                sim = float(np.dot(query_feat, cand_feat.T)[0, 0])
                sim = max(-1.0, min(1.0, sim))
                results.append({
                    'index': idx,
                    'similarity': round(sim, 4),
                    'distance': round(1.0 - sim, 4),
                    'match_level': (
                        'identical' if sim >= 0.99 else
                        'high' if sim >= 0.90 else
                        'medium' if sim >= 0.70 else
                        'low' if sim >= 0.50 else 'none'
                    ),
                })
            except Exception:
                continue

        # 按相似度降序排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = results[:top_k]

        logger.info(
            f'批量对比完成: {len(results)}个候选 → top {len(top_results)}'
        )
        return {
            'matches': top_results,
            'query_dims': int(query_feat.shape[1]),
            'total_candidates': len(candidates),
            'valid_comparisons': len(results),
        }

    @staticmethod
    def load_image_thumbnail(image_data: bytes, max_size: tuple[int, int] = (224, 224)) -> bytes | None:
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
        # v2.2 扩充: 常见短表达/口语化好评
        '还行', '还可以', '挺好的', '不错啊', '很棒', '超级好',
        '太好了吧', '真好', '好极了', '妙', '绝', '正点', '地道', '正宗',
        '过瘾', '爽', '赞一个', '顶', '支持', '给个好评', '五星', '满分',
        '强烈推荐', '必买', '入手', '不亏', '血赚', '香', '真香',
        '爱了', '爱了爱了', 'yyds', '永远的神', 'YYDS',
        '到位', '讲究', '用心', '贴心', '周到', '细致', '到位了',
        '美', '美丽', '华丽', '炫酷', '酷', '时尚', '新潮', '潮流',
        '温和', '清爽', '滋润', '保湿', '服帖', '持久', '显色',
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
        # v2.2 扩充: 常见短表达/口语化差评
        '不太好', '不怎么样', '一般般', '马马虎虎', '凑合', '勉强',
        '没用', '没啥用', '废物', '就那样', '也就那样',
        '坑货', '雷', '踩雷', '翻车', '后悔', '血亏', '亏了', '上当了',
        '别买', '不要买', '千万别买', '避雷', '避坑', '拔草',
        '糊弄', '糊弄人', '蒙人', '骗人', '骗子', '诈骗',
        '异味', '臭味', '难闻', '刺鼻', '过敏', '烂脸', '刺激',
        '掉色', '缩水', '起球', '变形', '开胶', '断裂', '生锈',
        '无语', '醉了', '服了', '呵呵', '？？？', '？？',
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

        # 尝试用 jieba 分词 + 字符 n-gram 补充 (解决 jieba 拆分短语问题)
        # 例: "还行" → jieba: ["还","行"], n-gram: ["还行"] → 命中词典
        words = []
        try:
            import jieba
            words = list(jieba.cut(text))
            # 补充字符 n-gram (2-4字窗口), 捕获被 jieba 拆分的短语
            clean_text = text.strip()
            for win in [2, 3, 4]:
                for i in range(len(clean_text) - win + 1):
                    words.append(clean_text[i:i + win])
        except ImportError:
            # 按字符滑动窗口 + 单字
            words = list(text)
            for win in [2, 3, 4]:
                for i in range(len(text) - win + 1):
                    words.append(text[i:i + win])

        pos_count = 0
        neg_count = 0
        words_list = list(words)

        # 找到每个匹配词在原文中的位置（用于否定词检测）
        def _find_text_pos(word: str, start_from: int = 0) -> int:
            idx = text.find(word, start_from)
            return idx if idx != -1 else -1

        for i, w in enumerate(words_list):
            w_clean = w.strip()
            if not w_clean:
                continue

            if w_clean in FeatureExtractor._POSITIVE_WORDS:
                # 确定这个词在原文中的位置
                word_pos = _find_text_pos(w_clean)
                # 检查前面是否有否定词（从原文中检查前2-4个字符）
                negated = False
                if word_pos >= 0:
                    start = max(0, word_pos - 3)
                    prefix = text[start:word_pos]
                    for neg_word in FeatureExtractor._NEGATION_WORDS:
                        if neg_word in prefix:
                            negated = True
                            break
                if not negated:
                    # 回退到 words_list 位置检查
                    for j in range(max(0, i - 2), i):
                        if words_list[j].strip() in FeatureExtractor._NEGATION_WORDS:
                            negated = True
                            break
                if negated:
                    neg_count += 1
                else:
                    pos_count += 1

            elif w_clean in FeatureExtractor._NEGATIVE_WORDS:
                word_pos = _find_text_pos(w_clean)
                negated = False
                if word_pos >= 0:
                    start = max(0, word_pos - 3)
                    prefix = text[start:word_pos]
                    for neg_word in FeatureExtractor._NEGATION_WORDS:
                        if neg_word in prefix:
                            negated = True
                            break
                if not negated:
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
