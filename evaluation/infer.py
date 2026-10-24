"""Run greedy inference on a generated instance or an exact archived paper input.

Use --archive static|dynamic --list_cases to find learned-policy case IDs.
Use --inspect_fixture with --archive and --case for CPU-only input/model checks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from analysis.archive import Bank, digest
from src.HGT.inference_env import InferenceEnv
from src.data_generator import DataGenerator
from src.utils.config_util import create_inference_config, parse_size
from src.utils.model_utils import load_model
from src.utils.reproducibility import seed_context, set_global_seed
from verification.schedules import need

ROOT = Path(__file__).resolve().parents[1]


def numpy_instance(raw):
    """Decode exact saved data using the frozen evaluation's array conversion."""
    raw = copy.deepcopy(raw)
    for key, values in raw['static'].items():
        raw['static'][key] = [np.asarray(value, dtype=np.int64 if key == 'job_length' else np.float32)
                              for value in values]
    return raw


def archived_case(bank, case_id, model_path=None):
    """Resolve one recorded policy/input pair and its verified packaged model."""
    items = [item for item in bank.index['cases'] if item['id'] == case_id]
    need(len(items) == 1, 'Unknown archived case; use --list_cases for exact IDs')
    item = items[0]
    identity = item['checkpoint_source_sha256']
    need(identity is not None, 'This case is a dispatch reference, not a learned-policy case')
    record = bank.json(item['member'])
    need(record['task']['id'] == case_id and record['task']['fixture'] == item['fixture'],
         'Case identity differs from archive index')
    fixture_hash = bank.index['fixtures'][item['fixture']]['source_sha256']
    need(record['fixture_sha256'] == fixture_hash and record['checkpoint_sha256'] == identity,
         'Case input/checkpoint source identity differs from archive index')
    models = json.loads((ROOT / 'models/manifest.json').read_text())['models']
    candidates = sorted((name, meta) for name, meta in models.items() if meta['source_sha256'] == identity)
    need(bool(candidates), 'Archived checkpoint is absent from the model manifest')
    model_id, model = candidates[0]
    selected = ROOT / model['path']
    need(digest(selected) == model['sha256'], 'Packaged selected checkpoint hash mismatch')
    if model_path is not None:
        selected = Path(model_path)
        need(digest(selected) == model['sha256'],
             '--model does not match the selected checkpoint for this archived case')
    fixture = bank.json('fixtures.json')[item['fixture']]
    need(fixture['family'] in ['static', 'dynamic'], 'Unsupported archived input family')
    instance = numpy_instance(fixture['instance'])
    n_j, n_m = parse_size(fixture['size'])
    need(len(instance['static']['job_length'][0]) == n_j
         and instance['static']['op_pt'][0].shape == (sum(instance['static']['job_length'][0]), n_m),
         'Saved fixture dimensions differ from declared size')
    metadata = {
        'case': case_id, 'fixture': item['fixture'], 'fixture_source_sha256': fixture_hash,
        'fixture_payload_sha256': hashlib.sha256(json.dumps(fixture, sort_keys=True,
                                                          separators=(',', ':')).encode()).hexdigest(),
        'archive_sha256': bank.meta['sha256'], 'model_id': model_id,
        'checkpoint_source_sha256': identity, 'checkpoint_sha256': model['sha256'],
        'initial_jobs': n_j, 'machines': n_m,
        'runtime_static_arrays': {key: [{'shape': list(value.shape), 'dtype': str(value.dtype)}
                                      for value in values] for key, values in instance['static'].items()},
        'event_counts': {key: len(value) for key, value in instance['dynamic_events'].items()},
    }
    return selected, instance, n_j, n_m, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Checkpoint path; optional for an archived case, where it must match the selected policy')
    parser.add_argument('--archive', choices=['static', 'dynamic'], help='Replay an exact input from the checksum-verified results archive')
    parser.add_argument('--case', help='Exact archived learned-policy case ID')
    parser.add_argument('--list_cases', action='store_true', help='List replayable learned-policy case IDs in --archive')
    parser.add_argument('--inspect_fixture', action='store_true', help='Verify archived input and construct the selected model on CPU without inference')
    parser.add_argument('--size', default='20x10', help='Generated shop size, NjxNm (unused with --archive)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.archive:
        if args.list_cases and (args.case or args.inspect_fixture or args.model):
            parser.error('--list_cases cannot be combined with --case, --model or --inspect_fixture')
        if not args.list_cases and not args.case:
            parser.error('--archive requires --case or --list_cases')
    elif args.case or args.list_cases or args.inspect_fixture:
        parser.error('--case, --list_cases and --inspect_fixture require --archive')
    elif not args.model:
        parser.error('Generated inference requires --model')

    set_global_seed(args.seed)
    metadata = None
    if args.archive:
        bank = Bank(ROOT, args.archive)
        try:
            if args.list_cases:
                for item in bank.index['cases']:
                    if item['checkpoint_source_sha256'] is not None:
                        print(item['id'])
                return
            model_path, instance, n_j, n_m, metadata = archived_case(bank, args.case, args.model)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        finally:
            bank.close()
    else:
        model_path = args.model
        n_j, n_m = parse_size(args.size)

    device = 'cpu' if args.inspect_fixture else args.device
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        parser.error('Requested CUDA device is unavailable; select --device cpu explicitly')
    policy, hparams, weights = load_model(str(model_path), device)
    if args.inspect_fixture:
        metadata.update(status='verified_input_and_cpu_model', inference_executed=False, device='cpu',
                        hgt_hyperparams=hparams, objective_weights=weights)
        print(json.dumps(metadata, indent=2))
        return
    if not args.archive:
        config = create_inference_config(
            n_j=n_j, n_m=n_m, device=device, seed=args.seed,
            num_new_job_arrivals=0, num_machine_breakdowns=0,
            num_priority_changes=0)
        with seed_context(args.seed):
            instance = DataGenerator(config).sample_env(num_envs=1)
    else:
        print(json.dumps(metadata, sort_keys=True))
    runner = InferenceEnv(
        n_j, n_m, device,
        use_pt_nodes=bool(hparams.get('use_pt_nodes', True)),
        use_dynamic_event_features=bool(hparams.get('use_dynamic_event_features', True)))
    result = runner.run(
        policy, instance,
        makespan_weight=weights['makespan_weight'], twt_weight=weights['twt_weight'])
    print(f"objective={result['objective']:.6f}")
    print(f"makespan={result['makespan']:.6f}")
    print(f"twt={result['twt']:.6f}")


if __name__ == '__main__':
    main()
