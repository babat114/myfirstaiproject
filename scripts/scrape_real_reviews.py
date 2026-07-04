"""
真实中文评论数据收集与数据集扩充工具

策略: 不使用 AI 数据增强, 而是收集真实公开数据源。
数据来源:
  1. 合并现有真实数据集 (ChnSentiCorp, 公开 NLP 数据集)
  2. 从本地 CSV/JSON 导入
  3. 数据质量检查 (去重、类别均衡、长度过滤)

输出格式: CSV (text,label) — 与现有训练管道兼容

Usage:
  python scripts/scrape_real_reviews.py --check DATASET.csv          # 数据质量检查
  python scripts/scrape_real_reviews.py --merge FILE1.csv FILE2.csv  # 合并去重
  python scripts/scrape_real_reviews.py --balance DATASET.csv        # 类别均衡
  python scripts/scrape_real_reviews.py --split DATASET.csv --ratio 0.8  # 拆分训练/测试
"""
import sys
import os
import csv
import json
import hashlib
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'uploads' / 'datasets'


def load_csv(file_path: str) -> list:
    """Load a CSV file. Expects 'text' and 'label' columns."""
    rows = []
    if not os.path.exists(file_path):
        print(f"[ERROR] 文件不存在: {file_path}")
        return rows

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text', '').strip()
            label = row.get('label', '').strip()
            if text and label:
                rows.append({'text': text, 'label': label})

    return rows


def save_csv(rows: list, file_path: str):
    """Save rows to CSV."""
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存 {len(rows)} 条 → {file_path}")


def check_quality(file_path: str):
    """Analyze dataset quality."""
    rows = load_csv(file_path)
    if not rows:
        print("空数据集！")
        return

    print(f"\n{'=' * 60}")
    print(f"  数据质量检查: {file_path}")
    print(f"{'=' * 60}")

    # 基本统计
    labels = Counter(r['label'] for r in rows)
    lengths = [len(r['text']) for r in rows]

    print(f"\n  总条数: {len(rows)}")
    print(f"  类别分布:")
    for label, count in labels.most_common():
        pct = count / len(rows) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")

    imbalance_ratio = max(labels.values()) / min(labels.values()) if len(labels) > 1 else float('inf')
    print(f"  不平衡比例: {imbalance_ratio:.1f}:1")

    # 文本长度分布
    print(f"\n  文本长度:")
    print(f"    最短: {min(lengths)} 字")
    print(f"    最长: {max(lengths)} 字")
    print(f"    平均: {sum(lengths) / len(lengths):.1f} 字")
    print(f"    中位数: {sorted(lengths)[len(lengths)//2]} 字")

    short_count = sum(1 for l in lengths if l < 10)
    long_count = sum(1 for l in lengths if l > 500)
    print(f"    过短 (<10字): {short_count} 条")
    print(f"    过长 (>500字): {long_count} 条")

    # 去重检查
    text_hashes = set()
    duplicates = 0
    for r in rows:
        h = hashlib.md5(r['text'].encode('utf-8')).hexdigest()
        if h in text_hashes:
            duplicates += 1
        text_hashes.add(h)
    print(f"\n  重复文本: {duplicates} 条 ({duplicates / len(rows) * 100:.1f}%)")

    # 质量评估
    print(f"\n  质量评估:")
    issues = []
    if imbalance_ratio > 3:
        issues.append(f"类别严重不平衡 ({imbalance_ratio:.1f}:1), 建议用 --balance")
    if short_count > len(rows) * 0.1:
        issues.append(f"过多过短文本 ({short_count}条), 建议过滤")
    if duplicates > len(rows) * 0.05:
        issues.append(f"重复文本较多 ({duplicates}条), 建议去重")
    if len(rows) < 2000:
        issues.append(f"数据集偏小 ({len(rows)}条), 保守训练可缓解过拟合")

    if issues:
        for issue in issues:
            print(f"    [!] {issue}")
    else:
        print(f"    [OK] 数据质量良好")

    return {
        'total': len(rows),
        'labels': dict(labels),
        'imbalance_ratio': imbalance_ratio,
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'duplicates': duplicates,
        'issues': issues,
    }


def deduplicate(rows: list) -> tuple:
    """Remove duplicate texts."""
    seen = set()
    unique = []
    dup_count = 0
    for r in rows:
        h = hashlib.md5(r['text'].encode('utf-8')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(r)
        else:
            dup_count += 1
    return unique, dup_count


def filter_by_length(rows: list, min_len: int = 5, max_len: int = 500) -> list:
    """Filter out texts that are too short or too long."""
    filtered = [r for r in rows if min_len <= len(r['text']) <= max_len]
    removed = len(rows) - len(filtered)
    if removed:
        print(f"长度过滤: 移除 {removed} 条 (min={min_len}, max={max_len})")
    return filtered


def balance_classes(rows: list, method: str = 'downsample') -> list:
    """Balance class distribution.

    Methods:
      downsample: Reduce majority classes to match minority
      upsample: Duplicate minority classes to match majority (keeps real data)
    """
    label_groups = {}
    for r in rows:
        label_groups.setdefault(r['label'], []).append(r)

    if len(label_groups) <= 1:
        print("只有单一类别，无需均衡")
        return rows

    counts = {k: len(v) for k, v in label_groups.items()}
    print(f"原始分布: {counts}")

    if method == 'downsample':
        target = min(counts.values())
        balanced = []
        for label, group in label_groups.items():
            if len(group) > target:
                # 随机采样保持多样性
                sampled = random.sample(group, target)
                print(f"  {label}: {len(group)} → {target} (下采样)")
                balanced.extend(sampled)
            else:
                balanced.extend(group)
    elif method == 'upsample':
        target = max(counts.values())
        balanced = []
        for label, group in label_groups.items():
            if len(group) < target:
                # 重复采样 (保留所有真实数据 + 随机抽样补齐)
                extra_needed = target - len(group)
                extra = random.choices(group, k=extra_needed)
                print(f"  {label}: {len(group)} → {target} (上采样, +{extra_needed})")
                balanced.extend(group + extra)
            else:
                balanced.extend(group)

    random.shuffle(balanced)
    print(f"均衡后: {Counter(r['label'] for r in balanced)}")
    return balanced


def merge_files(file_paths: list, output_path: str, dedup: bool = True,
                balance: bool = False, min_len: int = 5, max_len: int = 500):
    """Merge multiple CSV files into one clean dataset."""
    all_rows = []
    sources = {}

    for fp in file_paths:
        rows = load_csv(fp)
        sources[fp] = len(rows)
        all_rows.extend(rows)
        print(f"加载: {len(rows)} 条 ({fp})")

    total_before = len(all_rows)
    print(f"\n合并前总计: {total_before} 条")

    # 去重
    if dedup:
        all_rows, dup_count = deduplicate(all_rows)
        if dup_count:
            print(f"去重: 移除 {dup_count} 条 (剩下 {len(all_rows)})")

    # 长度过滤
    all_rows = filter_by_length(all_rows, min_len, max_len)

    # 类别均衡
    if balance:
        all_rows = balance_classes(all_rows)

    # 随机打乱
    random.shuffle(all_rows)

    # 保存
    save_csv(all_rows, output_path)

    # 统计
    labels = Counter(r['label'] for r in all_rows)
    print(f"\n合并完成!")
    print(f"  总条数: {len(all_rows)} (原始: {total_before})")
    print(f"  类别: {dict(labels)}")
    print(f"  来源: {sources}")

    return output_path


def split_dataset(file_path: str, train_ratio: float = 0.8):
    """Split dataset into train and test CSVs."""
    rows = load_csv(file_path)
    random.shuffle(rows)

    # 分层抽样 (按标签)
    label_groups = {}
    for r in rows:
        label_groups.setdefault(r['label'], []).append(r)

    train_rows = []
    test_rows = []

    for label, group in label_groups.items():
        split_idx = int(len(group) * train_ratio)
        train_rows.extend(group[:split_idx])
        test_rows.extend(group[split_idx:])
        print(f"  {label}: {len(group)} → train={split_idx}, test={len(group) - split_idx}")

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    base = file_path.rsplit('.', 1)[0]
    train_path = f"{base}_train.csv"
    test_path = f"{base}_test.csv"

    save_csv(train_rows, train_path)
    save_csv(test_rows, test_path)

    print(f"\n拆分完成: train={len(train_rows)}, test={len(test_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description='真实中文评论数据收集与整理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/scrape_real_reviews.py --check shopping.csv
  python scripts/scrape_real_reviews.py --merge data1.csv data2.csv --output merged.csv
  python scripts/scrape_real_reviews.py --balance dataset.csv --method upsample
  python scripts/scrape_real_reviews.py --split dataset.csv --ratio 0.8
        """
    )
    parser.add_argument('--check', type=str, default=None,
                        help='检查数据集质量 (CSV 路径)')
    parser.add_argument('--merge', nargs='+', default=None,
                        help='合并多个 CSV 文件')
    parser.add_argument('--output', type=str, default=None,
                        help='合并/均衡后的输出路径')
    parser.add_argument('--balance', type=str, default=None,
                        help='均衡数据集的类别分布')
    parser.add_argument('--method', type=str, default='upsample',
                        choices=['upsample', 'downsample'],
                        help='均衡方法 (upsample: 保留所有真实数据; downsample: 减少多数类)')
    parser.add_argument('--split', type=str, default=None,
                        help='拆分数据集为训练/测试')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='训练集比例 (默认: 0.8)')
    parser.add_argument('--no-dedup', action='store_true',
                        help='跳过去重')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_quality(args.check)

    elif args.merge:
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(OUTPUT_DIR / f'merged_dataset_{timestamp}.csv')
        merge_files(args.merge, output_path, dedup=not args.no_dedup)

    elif args.balance:
        rows = load_csv(args.balance)
        balanced = balance_classes(rows, method=args.method)
        output_path = args.output
        if not output_path:
            base = args.balance.rsplit('.', 1)[0]
            output_path = f"{base}_balanced.csv"
        save_csv(balanced, output_path)

    elif args.split:
        split_dataset(args.split, args.ratio)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
