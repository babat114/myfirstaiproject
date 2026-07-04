#!/usr/bin/env python
"""
Re-evaluate All Models Against Independent Test Sets

Usage:
  python scripts/reevaluate_with_independent_tests.py --all --dry-run
  python scripts/reevaluate_with_independent_tests.py --all
  python scripts/reevaluate_with_independent_tests.py --tier-1-only
  python scripts/reevaluate_with_independent_tests.py --dataset real_car_eval
  python scripts/reevaluate_with_independent_tests.py --model-uuid <uuid>

Output:
  experiments/independent_eval_report.json — full results with generalization gaps
"""
import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models.model_record import ModelRecord
from app.models.dataset import Dataset
from app.services.inference_service import ModelInferenceService

REPORT_PATH = os.path.join(os.path.dirname(__file__), '..', 'experiments',
                          'independent_eval_report.json')


def evaluate_model_against_independent_test(model: ModelRecord) -> dict:
    """
    Evaluate one model against its linked independent test set.
    """
    result = {
        'model_id': model.id,
        'model_uuid': model.uuid,
        'model_name': model.name,
        'framework': model.framework,
        'original_accuracy': model.accuracy,
        'original_f1': model.f1_score,
        'training_dataset_id': model.training_dataset_id,
    }

    # Find independent test sets
    test_sets = Dataset.query.filter_by(
        source_dataset_id=model.training_dataset_id,
        is_test_set=True,
        status='ready'
    ).all()

    if not test_sets:
        result['status'] = 'no_test_set'
        result['note'] = 'No independent test set available'
        logger.info(f"No test set for model {model.name}")
        return result

    best_result = None
    best_gap = None

    for test_ds in test_sets:
        logger.info(f"Evaluating {model.name} against {test_ds.name}...")

        try:
            eval_result = ModelInferenceService.test_model_with_split(
                model, test_dataset=test_ds
            )
        except Exception as e:
            logger.error(f"Evaluation failed for {model.name}: {e}")
            eval_result = {'success': False, 'error': str(e)}

        if eval_result.get('success'):
            ind_acc = eval_result.get('accuracy')
            ind_f1 = eval_result.get('f1_macro') or eval_result.get('f1_weighted')
            orig_acc = model.accuracy or 0

            gap = round(orig_acc - ind_acc, 4) if ind_acc is not None else None

            eval_entry = {
                'test_dataset_id': test_ds.id,
                'test_dataset_name': test_ds.name,
                'test_dataset_uuid': test_ds.uuid,
                'collection_method': test_ds.collection_method,
                'independent_accuracy': ind_acc,
                'independent_f1': ind_f1,
                'generalization_gap': gap,
                'num_samples': eval_result.get('num_samples'),
            }

            # Save to model record
            ind_metrics = {
                'ind_test_accuracy': ind_acc,
                'ind_test_f1_weighted': eval_result.get('f1_weighted'),
                'ind_test_f1_macro': eval_result.get('f1_macro'),
                'ind_test_precision_macro': eval_result.get('precision_macro'),
                'ind_test_recall_macro': eval_result.get('recall_macro'),
                'test_dataset_name': test_ds.name,
                'test_dataset_uuid': test_ds.uuid,
                'collection_method': test_ds.collection_method,
                'evaluated_at': datetime.now(timezone.utc).isoformat(),
            }
            model.set_independent_metrics(ind_metrics)
            model.independent_test_dataset_id = test_ds.id

            # Track best (closest to original) result
            if gap is not None and (best_gap is None or abs(gap) < abs(best_gap)):
                best_gap = gap
                best_result = eval_entry

            logger.info(f"  Independent acc={ind_acc}, gap={gap}")

        else:
            eval_entry = {
                'test_dataset_id': test_ds.id,
                'test_dataset_name': test_ds.name,
                'error': eval_result.get('error', 'Unknown error'),
            }
            logger.warning(f"  Failed: {eval_entry['error']}")

    if best_result:
        result['status'] = 'evaluated'
        result.update(best_result)
    else:
        result['status'] = 'eval_failed'
        result['note'] = 'All test sets failed evaluation'

    return result


def reevaluate_all(dry_run: bool = False, tier1_only: bool = False,
                   dataset_key: str = None, model_uuid: str = None):
    """Main re-evaluation loop."""
    app = create_app()
    with app.app_context():
        query = ModelRecord.query.filter(
            ModelRecord.status.in_(['trained', 'deployed']),
            ModelRecord.model_file_path.isnot(None),
            ModelRecord.accuracy.isnot(None),
        )

        if model_uuid:
            query = query.filter_by(uuid=model_uuid)
        elif dataset_key:
            # Find models trained on datasets matching the key
            ds_ids = [d.id for d in Dataset.query.filter(
                Dataset.file_path.contains(dataset_key)
            ).all()]
            if ds_ids:
                query = query.filter(ModelRecord.training_dataset_id.in_(ds_ids))
            else:
                logger.error(f"No datasets found matching key: {dataset_key}")
                return

        if tier1_only:
            # Only models with perfect accuracy (likely overfit)
            query = query.filter(ModelRecord.accuracy >= 0.999)

        models = query.order_by(ModelRecord.accuracy.desc()).all()
        logger.info(f"Found {len(models)} models to re-evaluate")

        results = []
        ok_count, no_test_count, fail_count = 0, 0, 0

        for model in models:
            logger.info(f"\n--- {model.name} (acc={model.accuracy}) ---")
            r = evaluate_model_against_independent_test(model)
            results.append(r)

            if r['status'] == 'evaluated':
                ok_count += 1
                if not dry_run:
                    try:
                        db.session.commit()
                    except Exception as e:
                        db.session.rollback()
                        logger.error(f"DB commit failed: {e}")
            elif r['status'] == 'no_test_set':
                no_test_count += 1
            else:
                fail_count += 1

        # Generate report
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_models': len(models),
            'evaluated': ok_count,
            'no_test_set': no_test_count,
            'eval_failed': fail_count,
            'dry_run': dry_run,
            'results': results,
            # Summary: generalization gaps for evaluated models
            'summary': {
                'models_with_large_gap': [
                    r for r in results
                    if r.get('generalization_gap') is not None and abs(r['generalization_gap']) > 0.15
                ],
                'models_with_moderate_gap': [
                    r for r in results
                    if r.get('generalization_gap') is not None and 0.05 < abs(r['generalization_gap']) <= 0.15
                ],
                'avg_gap': round(
                    sum(abs(r['generalization_gap']) for r in results
                        if r.get('generalization_gap') is not None) /
                    max(1, sum(1 for r in results if r.get('generalization_gap') is not None)),
                    4
                ) if any(r.get('generalization_gap') is not None for r in results) else None,
            }
        }

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"Re-evaluation complete: {ok_count} OK, {no_test_count} no test set, {fail_count} failed")
        logger.info(f"Avg generalization gap: {report['summary']['avg_gap']}")
        logger.info(f"Large gap (>15%): {len(report['summary']['models_with_large_gap'])}")
        logger.info(f"Moderate gap (5-15%): {len(report['summary']['models_with_moderate_gap'])}")
        logger.info(f"Report saved to: {REPORT_PATH}")

        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-evaluate models against independent test sets')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview only, do not save to DB')
    parser.add_argument('--all', action='store_true',
                       help='Re-evaluate all trained models')
    parser.add_argument('--tier-1-only', action='store_true',
                       help='Only re-evaluate models with perfect accuracy')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Only models trained on datasets matching this key')
    parser.add_argument('--model-uuid', type=str, default=None,
                       help='Evaluate a specific model by UUID')
    args = parser.parse_args()

    if not any([args.all, args.tier_1_only, args.dataset, args.model_uuid]):
        logger.info("No scope specified. Running --tier-1-only as default...")
        args.tier_1_only = True

    reevaluate_all(
        dry_run=args.dry_run,
        tier1_only=args.tier_1_only,
        dataset_key=args.dataset,
        model_uuid=args.model_uuid,
    )
