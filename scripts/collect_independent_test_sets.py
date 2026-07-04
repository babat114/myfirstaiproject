#!/usr/bin/env python
"""
Collect Independent Test Sets for All Datasets

Usage:
  python scripts/collect_independent_test_sets.py --dry-run
  python scripts/collect_independent_test_sets.py --all
  python scripts/collect_independent_test_sets.py --dataset real_car_eval
  python scripts/collect_independent_test_sets.py --tier-1-only
  python scripts/collect_independent_test_sets.py --method synthetic

Workflow:
  1. For each training dataset, look up independent test source in registry
  2. If found: download from URL / cross-reference local dataset
  3. If not found: generate synthetic test data via TestDataGenerator
  4. Validate, save CSV, register as Dataset with is_test_set=True
  5. Output collection report
"""
import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.dataset import Dataset
from app._timezone import localnow
from scripts._independent_test_registry import get_independent_source

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'datasets')
REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments',
                          'independent_test_collection_report.json')


def ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


def download_from_url(url: str, timeout: int = 30) -> bytes | None:
    """Download file from URL with retries."""
    import urllib.request
    import urllib.error

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0 (compatible; MLPipeline/1.0)')
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1}/3 failed for {url}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


def parse_uci_data(content: bytes, target_col: str) -> 'pd.DataFrame | None':
    """Parse UCI .data format (comma or whitespace delimited, no header)."""
    import pandas as pd
    import io

    try:
        text = content.decode('utf-8', errors='replace')
        # Try comma first, then whitespace
        for sep in [',', r'\s+']:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep, header=None, engine='python')
                if df.shape[1] >= 2:
                    # Check if last column could be target
                    logger.info(f"Parsed UCI data: {df.shape} with sep={sep}")
                    return df
            except Exception:
                continue
        return None
    except Exception as e:
        logger.error(f"Failed to parse UCI data: {e}")
        return None


def parse_csv_content(content: bytes) -> 'pd.DataFrame | None':
    """Parse CSV content to DataFrame."""
    import pandas as pd
    import io

    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
        try:
            text = content.decode(encoding, errors='replace')
            df = pd.read_csv(io.StringIO(text))
            if len(df) > 0:
                logger.info(f"Parsed CSV: {df.shape} with encoding={encoding}")
                return df
        except Exception:
            continue
    return None


def collect_url_source(dataset: Dataset, source: dict) -> 'pd.DataFrame | None':
    """Download and parse independent test data from URL."""
    urls = source.get('urls', [source.get('url')]) if 'urls' in source else [source.get('url')]
    urls = [u for u in urls if u]  # filter None

    for url in urls:
        logger.info(f"Downloading from {url}...")
        content = download_from_url(url)
        if content is None:
            continue

        # Try CSV parsing first
        df = parse_csv_content(content)
        if df is not None and len(df) >= 20:
            return df

        # Try UCI .data format
        target = source.get('target', 'class')
        df = parse_uci_data(content, target)
        if df is not None and len(df) >= 20:
            return df

    logger.warning(f"All URL sources failed for {dataset.name}")
    return None


def collect_local_cross_domain(dataset: Dataset, source: dict) -> 'pd.DataFrame | None':
    """Use another local dataset as cross-domain test set."""
    local_key = source.get('local_key')
    if not local_key:
        return None

    local_path = os.path.join(OUTPUT_DIR, f'{local_key}.csv')
    if not os.path.exists(local_path):
        logger.warning(f"Cross-domain source not found: {local_path}")
        return None

    import pandas as pd
    try:
        df = pd.read_csv(local_path)
        logger.info(f"Loaded cross-domain source: {local_path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Failed to load cross-domain source: {e}")
        return None


def collect_synthetic(dataset: Dataset) -> 'pd.DataFrame | None':
    """Generate synthetic independent test data from training distribution."""
    import pandas as pd
    from app.utils.test_data_generator import TestDataGenerator

    if not os.path.exists(dataset.file_path):
        logger.warning(f"Training data not found: {dataset.file_path}")
        return None

    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return None

    # Determine target column
    target_col = None
    if dataset.summary_json:
        try:
            summary = json.loads(dataset.summary_json)
            target_col = summary.get('target_column')
        except Exception:
            pass
    if not target_col:
        target_col = df.columns[-1]

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not in dataset {dataset.name}")
        return None

    # Determine task type from category
    task_map = {
        'classification': 'classification',
        'regression': 'regression',
        'clustering': 'clustering',
        'nlp': 'nlp',
        'tabular': 'classification',  # default to classification
        'biology': 'classification',
        'finance': 'classification',
    }
    task_type = task_map.get(dataset.category, 'classification')

    logger.info(f"Generating synthetic test data for {dataset.name} (task={task_type})...")
    try:
        test_df = TestDataGenerator.from_training_data(
            df, target_col, task_type,
            n_samples=min(len(df), 500),
            perturbation=0.05
        )
        return test_df
    except Exception as e:
        logger.error(f"Synthetic generation failed: {e}")
        return None


def validate_and_save(test_df, dataset: Dataset, source: dict, collection_method: str,
                      dry_run: bool = False) -> str | None:
    """
    Validate test data and save to disk.

    Returns the file path if successful, None otherwise.
    """
    import pandas as pd
    from app.utils.test_data_generator import TestDataGenerator

    if len(test_df) < 20:
        logger.warning(f"Test set too small ({len(test_df)} rows), skipping")
        return None

    # Load training data for validation
    try:
        train_df = pd.read_csv(dataset.file_path)
    except Exception:
        train_df = test_df  # skip validation if can't load training data

    # Determine target column
    target_col = source.get('target', train_df.columns[-1] if len(train_df.columns) > 0 else test_df.columns[-1])

    # If target is not in test_df but is in train_df, use test_df's last column
    if target_col not in test_df.columns:
        target_col = test_df.columns[-1]
        logger.info(f"Using last column as target: '{target_col}'")

    # Validate
    validation = TestDataGenerator.validate_test_data(test_df, train_df, target_col)
    if validation['issues']:
        for issue in validation['issues']:
            logger.warning(f"  Issue: {issue}")
        if any('missing' in i.lower() for i in validation['issues']):
            return None  # Can't use if features are missing

    for warning in validation.get('warnings', []):
        logger.info(f"  Warning: {warning}")

    if dry_run:
        logger.info(f"[DRY RUN] Would save test set: {len(test_df)} rows")
        return "DRY_RUN"

    # Generate output filename
    basename = os.path.basename(dataset.file_path).replace('.csv', '')
    out_name = f'ind_test_{basename}_{collection_method}.csv'
    out_path = os.path.join(OUTPUT_DIR, out_name)

    # Ensure target column is present
    if target_col not in test_df.columns and target_col in train_df.columns:
        test_df[target_col] = pd.Series(dtype=train_df[target_col].dtype)

    test_df.to_csv(out_path, index=False)
    logger.info(f"Saved test set: {out_path} ({len(test_df)} rows)")
    return out_path


def register_test_dataset(dataset: Dataset, test_path: str, source: dict,
                          collection_method: str, test_df) -> Dataset | None:
    """Register the test dataset in the database."""
    import pandas as pd

    test_name = f"{dataset.name} [IndTest-{collection_method}]"

    # Check for existing test set with same source
    existing = Dataset.query.filter_by(
        source_dataset_id=dataset.id,
        is_test_set=True,
        collection_method=collection_method
    ).first()
    if existing:
        logger.info(f"Test dataset already exists: {existing.name} (id={existing.id})")
        # Update file path if changed
        if existing.file_path != test_path:
            existing.file_path = test_path
            existing.file_size = os.path.getsize(test_path) if os.path.exists(test_path) else 0
            existing.row_count = len(test_df)
            existing.column_count = len(test_df.columns)
            db.session.commit()
        return existing

    # Determine target column
    target_col = source.get('target', test_df.columns[-1])

    # Build summary JSON
    summary = {
        'target_column': target_col,
        'columns': list(test_df.columns),
        'dtypes': {str(k): str(v) for k, v in test_df.dtypes.items()},
        'n_samples': len(test_df),
        'n_features': len(test_df.columns) - 1,
    }

    # Determine format from file extension
    fmt = test_path.rsplit('.', 1)[-1] if '.' in test_path else 'csv'

    test_ds = Dataset(
        name=test_name,
        description=f'Independent test set for {dataset.name} — {source.get("note", "")}',
        file_path=test_path,
        file_size=os.path.getsize(test_path),
        file_format=fmt,
        category=dataset.category,
        row_count=len(test_df),
        column_count=len(test_df.columns),
        summary_json=json.dumps(summary, ensure_ascii=False),
        status='ready',
        is_public=False,
        is_test_set=True,
        source_dataset_id=dataset.id,
        collection_method=collection_method,
        owner_id=1,  # admin
    )

    db.session.add(test_ds)
    db.session.commit()
    logger.info(f"Registered test dataset: {test_ds.name} (id={test_ds.id})")
    return test_ds


def collect_for_dataset(dataset: Dataset, dry_run: bool = False,
                        method_filter: str = None, force: bool = False) -> dict:
    """
    Collect independent test data for one dataset.

    Returns: {'status': 'ok'|'skipped'|'failed', 'collection_method': str, ...}
    """
    filename = os.path.basename(dataset.file_path)
    source = get_independent_source(filename)

    result = {
        'dataset_id': dataset.id,
        'dataset_name': dataset.name,
        'filename': filename,
        'category': dataset.category,
        'has_url_source': source is not None,
        'source_tier': source.get('tier') if source else None,
    }

    # Skip if already has test sets and not forcing re-collection
    if not force:
        existing_count = Dataset.query.filter_by(
            source_dataset_id=dataset.id, is_test_set=True
        ).count()
        if existing_count > 0:
            # But if we're trying a better source (url vs synthetic), allow upgrade
            existing_url = Dataset.query.filter_by(
                source_dataset_id=dataset.id, is_test_set=True,
                collection_method='url'
            ).count()
            if existing_url > 0:
                logger.info(f"Skipping {dataset.name}: already has URL test set")
                result['status'] = 'skipped'
                result['reason'] = f'already has URL test set'
                return result
            elif method_filter == 'url' and source and source.get('tier') in (1, 3):
                # Upgrade: synthetic exists but URL source available — proceed
                logger.info(f"Upgrading {dataset.name}: replacing synthetic with URL source")
            else:
                logger.info(f"Skipping {dataset.name}: already has {existing_count} test set(s)")
                result['status'] = 'skipped'
                result['reason'] = f'already has {existing_count} test set(s)'
                return result

    # If force mode and URL, delete old synthetic test sets
    if force and method_filter == 'url':
        old_synthetic = Dataset.query.filter_by(
            source_dataset_id=dataset.id, is_test_set=True,
            collection_method='synthetic'
        ).all()
        for old in old_synthetic:
            logger.info(f"Removing old synthetic test set: {old.name}")
            db.session.delete(old)
        db.session.commit()

    test_df = None
    collection_method = None

    # Try URL source first (if not filtered)
    if source and source.get('source') == 'url' and method_filter in (None, 'url'):
        if method_filter != 'synthetic':
            test_df = collect_url_source(dataset, source)
            collection_method = 'url'
            if test_df is not None:
                logger.info(f"URL source succeeded for {dataset.name}")

    # Try local cross-domain
    if test_df is None and source and source.get('source') == 'local' and method_filter in (None, 'url'):
        if method_filter != 'synthetic':
            test_df = collect_local_cross_domain(dataset, source)
            collection_method = 'cross_domain'
            if test_df is not None:
                logger.info(f"Cross-domain source succeeded for {dataset.name}")

    # Fall back to synthetic
    if test_df is None and method_filter in (None, 'synthetic'):
        test_df = collect_synthetic(dataset)
        collection_method = 'synthetic'
        if test_df is not None:
            logger.info(f"Synthetic generation succeeded for {dataset.name}")

    if test_df is None:
        result['status'] = 'failed'
        result['reason'] = 'all collection methods failed'
        return result

    # Validate and save
    test_path = validate_and_save(test_df, dataset, source or {}, collection_method, dry_run)
    if test_path is None:
        result['status'] = 'failed'
        result['reason'] = 'validation failed'
        return result

    if dry_run:
        result['status'] = 'ok_dry_run'
        result['collection_method'] = collection_method
        result['test_rows'] = len(test_df)
        return result

    # Register in DB
    test_ds = register_test_dataset(dataset, test_path, source or {}, collection_method, test_df)
    if test_ds:
        result['status'] = 'ok'
        result['collection_method'] = collection_method
        result['test_dataset_id'] = test_ds.id
        result['test_dataset_uuid'] = test_ds.uuid
        result['test_rows'] = len(test_df)
    else:
        result['status'] = 'failed'
        result['reason'] = 'db registration failed'

    return result


def collect_all(dry_run: bool = False, method_filter: str = None,
                tier1_only: bool = False, dataset_key: str = None,
                force: bool = False):
    """Main collection loop."""

    app = create_app()
    with app.app_context():
        # Fetch all ready training datasets
        query = Dataset.query.filter_by(status='ready', is_test_set=False)
        if dataset_key:
            # Filter by filename containing key
            query = query.filter(Dataset.file_path.contains(dataset_key))
        datasets = query.order_by(Dataset.id).all()

        logger.info(f"Found {len(datasets)} training datasets")

        if tier1_only:
            # Filter to datasets with known URL sources
            tier1_keys = set()
            from scripts._independent_test_registry import EXACT_MATCH_SOURCES, NLP_SOURCES
            tier1_keys.update(EXACT_MATCH_SOURCES.keys())
            tier1_keys.update(NLP_SOURCES.keys())
            datasets = [d for d in datasets
                       if os.path.basename(d.file_path).replace('.csv', '') in tier1_keys]
            logger.info(f"Filtered to {len(datasets)} Tier-1 datasets")

        results = []
        ok_count, fail_count, skip_count = 0, 0, 0

        for ds in datasets:
            logger.info(f"\n--- Processing: {ds.name} ({ds.category}) ---")
            try:
                r = collect_for_dataset(ds, dry_run=dry_run, method_filter=method_filter, force=force)
                results.append(r)

                if r['status'] == 'ok':
                    ok_count += 1
                elif r['status'] == 'ok_dry_run':
                    ok_count += 1
                elif r['status'] == 'skipped':
                    skip_count += 1
                else:
                    fail_count += 1

                # Rate limit for URL downloads
                if r.get('collection_method') == 'url':
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing {ds.name}: {e}", exc_info=True)
                results.append({'dataset_id': ds.id, 'dataset_name': ds.name,
                               'status': 'failed', 'reason': str(e)})
                fail_count += 1

        # Generate report
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_datasets': len(datasets),
            'ok': ok_count,
            'failed': fail_count,
            'skipped': skip_count,
            'dry_run': dry_run,
            'method_filter': method_filter,
            'results': results,
        }

        ensure_output_dirs()
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"Collection complete: {ok_count} OK, {fail_count} failed, {skip_count} skipped")
        logger.info(f"Report saved to: {REPORT_PATH}")

        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect independent test sets')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview only, do not save anything')
    parser.add_argument('--all', action='store_true',
                       help='Process all training datasets')
    parser.add_argument('--tier-1-only', action='store_true',
                       help='Only process datasets with known URL sources')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Process a specific dataset by key (e.g. real_car_eval)')
    parser.add_argument('--method', type=str, choices=['url', 'synthetic'],
                       default=None,
                       help='Only use this collection method')
    parser.add_argument('--force', action='store_true',
                       help='Force re-collection, replacing existing test sets')
    args = parser.parse_args()

    if not args.all and not args.dataset and not args.tier_1_only:
        logger.info("No scope specified. Use --all, --tier-1-only, or --dataset <key>")
        logger.info("Running with --tier-1-only as default...")
        args.tier_1_only = True

    collect_all(
        dry_run=args.dry_run,
        method_filter=args.method,
        tier1_only=args.tier_1_only,
        dataset_key=args.dataset,
        force=args.force,
    )
