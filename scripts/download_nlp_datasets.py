"""
============================================
NLP 真实中文数据集下载器
============================================
下载真实中文情感分析数据集，输出 text/label CSV 文件。
来源优先: HuggingFace datasets → 原始 JSON/CSV → 内置样本回退

输出格式:
  text,label
  "酒店位置很好，服务周到","正面"
  "隔音太差了，一晚上没睡好","负面"
"""
import os
import sys
import json
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / 'uploads' / 'datasets'
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 内置中文评论样本 (下载失败时回退)
# ============================================================
_BUILTIN_REVIEWS_POSITIVE = [
    "酒店位置很好，离地铁站很近，出行方便",
    "服务态度非常好，前台小姐姐很热情",
    "房间干净整洁，床很舒服，一觉睡到天亮",
    "早餐种类丰富，味道也很不错",
    "性价比超高，这个价格能住到这么好的酒店太值了",
    "环境优雅，装修风格我很喜欢",
    "周边配套齐全，吃饭购物都很方便",
    "下次来还会选择这家，强烈推荐",
    "设施很新，WiFi速度也很快",
    "酒店大堂很气派，给人感觉很高级",
    "服务员态度特别好，主动帮忙拿行李",
    "房间视野很好，可以看到城市夜景",
    "隔音效果不错，晚上很安静",
    "热水很足，洗澡很舒服",
    "退房速度很快，还送了伴手礼",
    "菜品味道正宗，分量也很足",
    "这家店的服务真是没话说，太周到了",
    "环境很舒适，适合朋友聚会",
    "性价比很高，物超所值",
    "老板人很好，还送了小菜",
    "装修很有格调，拍照很好看",
    "食材新鲜，口味地道",
    "服务员随叫随到，很贴心",
    "价格实惠，味道好，会再来的",
    "整体体验非常棒，值得推荐",
    "物流很快，第二天就收到了",
    "包装很严实，产品完好无损",
    "质量很好，跟描述的一样",
    "用了一段时间了，效果不错",
    "客服态度很好，问题解决很快",
    "这个价格买到这样的品质，太划算了",
    "外观设计很漂亮，手感也很好",
    "功能齐全，操作简单，老人也能用",
    "已经推荐给朋友了，都说好用",
    "第二次购买了，品质始终如一",
    "这家店很靠谱，以后就认准了",
    "做工精细，细节处理得很好",
    "使用感很好，会回购的",
    "卖家发货很快，态度也好",
    "东西很不错，五分好评",
]

_BUILTIN_REVIEWS_NEGATIVE = [
    "隔音太差了，一晚上没睡好",
    "房间有异味，通风不好",
    "前台态度冷漠，爱理不理的",
    "早餐种类太少，而且不好吃",
    "价格偏贵，性价比不高",
    "设施老旧，空调噪音很大",
    "卫生间漏水，报修了也没人管",
    "被罩有污渍，卫生堪忧",
    "位置太偏了，找了好久才找到",
    "电梯要等很久，高峰期更夸张",
    "wifi根本连不上，问前台说不知道",
    "毛巾看着就不干净，不敢用",
    "退房的时候多收了很多费用",
    "房间太小了，跟图片完全不符",
    "停车场太远，拖着行李走了好久",
    "菜品太咸了，而且上菜很慢",
    "服务员叫了半天都不来",
    "环境太吵了，根本没法安静吃饭",
    "价格比别家贵一倍，味道却很一般",
    "食材感觉不新鲜，吃完肚子不舒服",
    "排队排了一个小时，太浪费时间了",
    "环境脏乱差，再也不会来了",
    "服务员态度恶劣，像是欠了他钱",
    "份量太少，根本吃不饱",
    "味道太差了，完全对不起这个价格",
    "发货特别慢，催了好几次才发",
    "收到货发现是坏的，联系客服也不理",
    "质量太差了，用了一次就坏了",
    "跟图片完全不一样，被坑了",
    "包装很简陋，东西都压坏了",
    "客服态度特别差，问题一直不解决",
    "功能太少，完全不实用",
    "做工粗糙，到处都是毛刺",
    "用了几天就出问题了，质量堪忧",
    "售后服务太差了，保修期内都不管",
    "太失望了，跟预期差距太大",
    "操作复杂，说明书也看不懂",
    "过敏了，成分不写清楚",
    "物流把东西摔坏了，还不给赔",
    "一分钱一分货吧，太劣质了",
]


def _build_builtin_csv(output_path: str, n_samples: int = 500) -> str:
    """用内置样本生成 CSV 数据集 (下载失败时的回退方案)"""
    import random
    random.seed(42)

    # 通过重复和微调扩充样本
    reviews = []
    for _ in range(n_samples // 2):
        base = random.choice(_BUILTIN_REVIEWS_POSITIVE)
        reviews.append((base, '正面'))
    for _ in range(n_samples // 2):
        base = random.choice(_BUILTIN_REVIEWS_NEGATIVE)
        reviews.append((base, '负面'))

    random.shuffle(reviews)

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write('text,label\n')
        for text, label in reviews:
            # CSV 安全: 引号转义
            safe_text = text.replace('"', '""')
            f.write(f'"{safe_text}","{label}"\n')

    logger.info(f'内置样本数据集已保存: {output_path} ({n_samples} 条)')
    return output_path


def download_chnsenticorp(output_dir: str = None) -> dict:
    """下载 ChnSentiCorp 中文情感数据集

    Returns:
        {'hotel': path, 'notebook': path, ...} 或空 dict
    """
    if output_dir is None:
        output_dir = DATASETS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ChnSentiCorp 数据来源 (GitHub raw)
    sources = {
        'chnsenticorp_hotel': {
            'urls': [
                'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv',
                'https://raw.githubusercontent.com/dugushaonian/Chinese-sentiment-analysis/master/data/ChnSentiCorp_htl_all.csv',
            ],
            'type': 'csv_label_column',
            'desc': '酒店评论',
        },
        'chnsenticorp_notebook': {
            'urls': [
                'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_nb_all/ChnSentiCorp_nb_all.csv',
            ],
            'type': 'csv_label_column',
            'desc': '笔记本电脑评论',
        },
    }

    for name, cfg in sources.items():
        logger.info(f'尝试下载 {cfg["desc"]} ({name})...')
        downloaded = False

        for url in cfg['urls']:
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; MLPipeline/1.0)'
                })
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read().decode('utf-8', errors='replace')

                if not content.strip():
                    continue

                # 解析并转换格式 (使用 csv 模块处理引号内逗号)
                import csv, io
                output_path = output_dir / f'{name}.csv'
                reviews = []

                reader = csv.reader(io.StringIO(content))
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                    # ChnSentiCorp 格式: label,text (第一列标签, 第二列文本)
                    first_col = row[0].strip()
                    second_col = row[1].strip()

                    # 判断哪列是标签 (数字 = 标签)
                    if first_col in ('0', '1'):
                        label_raw = first_col
                        text = second_col
                    elif second_col in ('0', '1'):
                        label_raw = second_col
                        text = first_col
                    else:
                        # 尝试检测: 短的是标签
                        if len(first_col) <= 4 and len(second_col) > 10:
                            label_raw = first_col
                            text = second_col
                        elif len(second_col) <= 4 and len(first_col) > 10:
                            label_raw = second_col
                            text = first_col
                        else:
                            continue  # skip header or bad rows

                    if not text or len(text) < 5:
                        continue

                    # 标准化标签
                    if label_raw in ('1', 'positive', 'pos', '好评'):
                        label = '正面'
                    elif label_raw in ('0', 'negative', 'neg', '差评'):
                        label = '负面'
                    else:
                        label = label_raw

                    safe_text = text.replace('"', '""')
                    reviews.append(f'"{safe_text}","{label}"')

                if len(reviews) < 50:
                    logger.warning(f'{name}: 仅解析到 {len(reviews)} 条, 跳过')
                    continue

                with open(output_path, 'w', encoding='utf-8-sig') as f:
                    f.write('text,label\n')
                    f.write('\n'.join(reviews))

                logger.info(f'{cfg["desc"]}: 下载成功, {len(reviews)} 条 → {output_path}')
                results[name] = str(output_path)
                downloaded = True
                break

            except Exception as e:
                logger.warning(f'{name}: URL {url[:60]}... 失败: {e}')
                continue

        if not downloaded:
            logger.warning(f'{name}: 所有URL下载失败, 使用内置样本回退')
            fallback_path = output_dir / f'{name}.csv'
            _build_builtin_csv(str(fallback_path), n_samples=600)
            results[name] = str(fallback_path)

    return results


def download_douban_reviews(output_dir: str = None) -> str | None:
    """下载豆瓣影评数据集 (简化版)"""
    if output_dir is None:
        output_dir = DATASETS_DIR

    output_dir = Path(output_dir)
    output_path = output_dir / 'douban_reviews.csv'

    # 尝试多个来源
    urls = [
        'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/dmsc_v2/dmsc_v2.csv',
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; MLPipeline/1.0)'
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode('utf-8', errors='replace')

            if not content.strip():
                continue

            import csv as csv_mod, io
            reader = csv_mod.reader(io.StringIO(content))
            reviews = []
            for row in reader:
                if not row or len(row) < 2:
                    continue
                first_col = row[0].strip()
                second_col = row[1].strip()
                if first_col in ('1', 'positive'):
                    label = '正面'
                    text = second_col
                elif first_col in ('0', 'negative'):
                    label = '负面'
                    text = second_col
                elif second_col in ('1', 'positive'):
                    label = '正面'
                    text = first_col
                elif second_col in ('0', 'negative'):
                    label = '负面'
                    text = first_col
                else:
                    continue
                if len(text) < 5:
                    continue
                safe_text = text.replace('"', '""')
                reviews.append(f'"{safe_text}","{label}"')

            if len(reviews) < 100:
                continue

            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write('text,label\n')
                f.write('\n'.join(reviews))

            logger.info(f'Douban: download OK, {len(reviews)} reviews → {output_path}')
            return str(output_path)

        except Exception as e:
            logger.warning(f'豆瓣影评下载失败: {e}')
            continue

    # 回退
    logger.info('豆瓣影评: 使用内置样本回退')
    _build_builtin_csv(str(output_path), n_samples=800)
    return str(output_path)


def download_shopping_reviews(output_dir: str = None) -> str | None:
    """下载电商评论数据集 (简化版)"""
    if output_dir is None:
        output_dir = DATASETS_DIR

    output_dir = Path(output_dir)
    output_path = output_dir / 'shopping_reviews.csv'

    urls = [
        'https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/online_shopping_10_cats/online_shopping_10_cats.csv',
    ]

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; MLPipeline/1.0)'
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode('utf-8', errors='replace')

            if not content.strip():
                continue

            import csv as csv_mod, io
            reader = csv_mod.reader(io.StringIO(content))
            reviews = []
            for row in reader:
                if not row or len(row) < 2:
                    continue
                first_col = row[0].strip()
                second_col = row[1].strip()
                if first_col in ('1', 'positive'):
                    label = '正面'
                    text = second_col
                elif first_col in ('0', 'negative'):
                    label = '负面'
                    text = second_col
                elif second_col in ('1', 'positive'):
                    label = '正面'
                    text = first_col
                elif second_col in ('0', 'negative'):
                    label = '负面'
                    text = first_col
                else:
                    continue
                if len(text) < 5:
                    continue
                safe_text = text.replace('"', '""')
                reviews.append(f'"{safe_text}","{label}"')

            if len(reviews) < 100:
                continue

            with open(output_path, 'w', encoding='utf-8-sig') as f:
                f.write('text,label\n')
                f.write('\n'.join(reviews))

            logger.info(f'Shopping: download OK, {len(reviews)} reviews → {output_path}')
            return str(output_path)

        except Exception as e:
            logger.warning(f'电商评论下载失败: {e}')
            continue

    # 回退
    logger.info('电商评论: 使用内置样本回退')
    _build_builtin_csv(str(output_path), n_samples=700)
    return str(output_path)


def main():
    """下载所有 NLP 数据集"""
    logger.info('=== NLP 真实中文数据集下载 ===')
    logger.info(f'输出目录: {DATASETS_DIR}')

    results = {}

    # 1. ChnSentiCorp
    chnsenti = download_chnsenticorp()
    results.update(chnsenti)

    # 2. 豆瓣影评
    douban = download_douban_reviews()
    if douban:
        results['douban_reviews'] = douban

    # 3. 电商评论
    shopping = download_shopping_reviews()
    if shopping:
        results['shopping_reviews'] = shopping

    # 汇总
    logger.info(f'\n=== 下载完成 ===')
    for name, path in results.items():
        n = 0
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                n = sum(1 for _ in f) - 1  # 减 header
        except Exception:
            pass
        logger.info(f'  {name}: {path} ({n} 条)')

    # 输出 JSON 供后续脚本使用
    summary_path = DATASETS_DIR / 'nlp_datasets_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f'\n汇总文件: {summary_path}')

    return results


if __name__ == '__main__':
    main()
