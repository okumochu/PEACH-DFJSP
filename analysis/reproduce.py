"""Verify retained inputs/schedules and independently recompute paper statistics.

Run from the release root with: uv run python -m analysis.reproduce --output outputs/reproduced
No model inference, training, data generation or solver is invoked.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np

from analysis.archive import Bank, PHASES
from analysis.statistics import paired
from analysis.ablation import calculate as ablation
from verification.schedules import verify as verify_schedule, need
from verification.dynamic import verify as verify_dynamic
from verification.variance import verify as verify_variance, depth_statistics



def equal_fields(actual, expected):
    """Compare recomputed fields, without treating a saved summary as an input."""
    for key, value in actual.items():
        need(key in expected, f'Saved summary lacks {key}')
        old = expected[key]
        if isinstance(value, bool) or value is None or isinstance(value, str):
            need(value == old, f'Summary mismatch: {key}')
        else:
            need(math.isclose(value, old, rel_tol=1e-10, abs_tol=1e-9),
                 f'Summary mismatch: {key}: {value} vs {old}')


def metric(rows):
    values = [r['objective'] for r in rows]
    seconds = [r['measured_seconds'] for r in rows]
    return {'n': len(rows), 'mean_objective': statistics.mean(values),
            'sample_sd_objective': statistics.stdev(values),
            'mean_seconds': statistics.mean(seconds), 'sample_sd_seconds': statistics.stdev(seconds)}


def load_cases(bank, verifier):
    fixtures = bank.json('fixtures.json')
    records = {}
    for item in bank.index['cases']:
        row = bank.json(item['member'])
        need(row['task']['id'] == item['id'] and row['task']['fixture'] == item['fixture'],
             'Case identity differs from index')
        need(row['fixture_sha256'] == bank.index['fixtures'][item['fixture']]['source_sha256'],
             'Case fixture source hash differs from index')
        need(row.get('checkpoint_sha256') == item.get('checkpoint_source_sha256'),
             'Case checkpoint source hash differs from index')
        need(item['id'] not in records, 'Duplicate task')
        verifier(row, fixtures[item['fixture']])
        records[item['id']] = row
    need(len({r['task']['fixture'] for r in records.values()}) == len(fixtures), 'Unused/missing fixture')
    return records, fixtures


def group_cases(records, count):
    groups = defaultdict(list)
    for row in records.values():
        groups[row['task']['group']].append(row)
    for name, rows in groups.items():
        rows.sort(key=lambda r: r['task']['index'])
        need([r['task']['index'] for r in rows] == list(range(count)), f'Case coverage: {name}')
    return groups


def static(bank):
    records, fixtures = load_cases(bank, verify_schedule)
    need(len(records) == 600 and len(fixtures) == 120, 'Static coverage must be 600 cases/120 inputs')
    cp = {}
    for item in bank.index['cp']:
        row = bank.json(item['result'])
        verify_schedule(row, fixtures[item['fixture']])
        need(row['solver_status'] in ['OPTIMAL', 'FEASIBLE'], 'Unusable CP reference')
        cp[item['fixture']] = row
    need(set(cp) == set(fixtures), 'CP input coverage')
    saved = {r['group']: r for r in bank.json('summary.json')['cells']}
    result = {}
    for name, rows in sorted(group_cases(records, 30).items()):
        references = [cp[r['task']['fixture']]['objective'] for r in rows]
        for row, reference in zip(rows, references):
            need(row['task']['reference']['cp_objective'] == reference, 'CP pairing changed')
        stats = metric(rows)
        comparison = paired([r['objective'] for r in rows], references)
        equal_fields({'n': stats['n'], 'mean_objective': stats['mean_objective'],
                      'sample_std': stats['sample_sd_objective'],
                      'mean_gap_percent_vs_cp': comparison['mean_paired_gap_percent'],
                      'sample_sd_gap_pp': comparison['sample_sd_gap_pp'],
                      'wins': comparison['wins'], 'ties': comparison['ties'], 'losses': comparison['losses']}, saved[name])
        result[name] = {**stats, 'versus_cp': comparison,
                        'mean_retained_cp_seconds': statistics.mean(cp[r['task']['fixture']]['solve_seconds'] for r in rows)}
    need(len(result) == 20, 'Static policy/size cell coverage')
    sizes = sorted({r['task']['fixture'].split('/')[1] for r in records.values()}, key=lambda s: int(s.split('x')[0]))
    table = []
    for size in sizes:
        cells = {name.split('/')[0]: value for name, value in result.items() if name.endswith('/' + size)}
        table.append({'test_size': size,
                      'gaps_percent': {name: value['versus_cp']['mean_paired_gap_percent'] for name, value in cells.items()},
                      'sample_sd_gap_pp': {name: value['versus_cp']['sample_sd_gap_pp'] for name, value in cells.items()},
                      'mean_fresh_policy_seconds': statistics.mean(value['mean_seconds'] for value in cells.values()),
                      'mean_retained_cp_seconds': next(iter(cells.values()))['mean_retained_cp_seconds']})
    return {'cases_verified': len(records), 'cp_schedules_verified': len(cp), 'groups': result, 'paper_table': table}


def dynamic(bank):
    records, fixtures = load_cases(bank, verify_dynamic)
    need(len(records) == 1080 and len(fixtures) == 60, 'Dynamic coverage must be 1080 cases/60 inputs')
    groups = group_cases(records, 30)
    saved = bank.json('summary.json')['scenarios']
    output = {}
    for scenario in sorted(saved):
        policy = groups[scenario + '/rl']
        stats = metric(policy)
        equal_fields(stats, saved[scenario]['policy'])
        values = [r['objective'] for r in policy]
        dispatch = {}
        for method, expected in saved[scenario]['dispatch'].items():
            rows = groups[scenario + '/' + method]
            need([r['task']['fixture'] for r in rows] == [r['task']['fixture'] for r in policy], 'Unpaired dynamic inputs')
            comparison = paired(values, [r['objective'] for r in rows])
            equal_fields(metric(rows), expected)
            equal_fields(comparison, expected['comparison'])
            dispatch[method] = {**metric(rows), 'comparison': comparison}
        compact = [bank.json('references/compact/' + r['task']['id'] + '.json') for r in policy]
        for row, reference in zip(policy, compact):
            need(row['task']['fixture'] == reference['task']['fixture'], 'Compact input pairing')
            verify_dynamic(reference, fixtures[row['task']['fixture']])
        comparison = paired(values, [r['objective'] for r in compact])
        equal_fields(comparison, saved[scenario]['compact']['comparison'])
        output[scenario] = {'policy': stats, 'dispatch': dispatch, 'versus_compact': comparison}
    return {'cases_verified': len(records), 'compact_schedules_verified': 60, 'scenarios': output}


def public(bank):
    records, fixtures = load_cases(bank, verify_schedule)
    need(len(records) == 1600 and len(fixtures) == 1200, 'Public coverage must be 1600 cases/1200 inputs')
    saved = bank.json('summary.json')
    groups = group_cases(records, 100)
    arrays = {}
    all_values, all_refs = [], defaultdict(list)
    output = {}
    for name, rows in sorted(groups.items()):
        references = defaultdict(list)
        for row in rows:
            task = row['task']; refs = bank.index['references'][task['id']]
            for label, array_label in [('cp', 'ortools'), ('daniel', 'daniel')]:
                member = refs['arrays'][array_label]
                if member not in arrays:
                    arrays[member] = np.load(bank.file(member), allow_pickle=False)
                    need(arrays[member].shape == (100, 2) and np.isfinite(arrays[member]).all(), 'Public array dimensions/values')
                value = float(arrays[member][task['index'], 0])
                need(value == task['reference'][array_label + '_objective'], 'Public reference pairing')
                references[label].append(value)
            compact = bank.json(refs['compact']['member'])
            need(compact['task']['fixture'] == task['fixture'], 'Compact fixture pairing')
            verify_schedule(compact, fixtures[task['fixture']])
            historical = bank.json(refs['historical']['member'])
            need(historical['problem'] == fixtures[task['fixture']]['problem'], 'Historical fixture pairing')
            references['compact'].append(compact['objective'])
            references['historical'].append(historical['paper_makespan'])
        stats = metric(rows)
        equal_fields(stats, saved['groups'][name])
        values = [r['objective'] for r in rows]
        comparisons = {}
        for label, refs in references.items():
            comparisons[label] = paired(values, refs)
            equal_fields(comparisons[label], saved['groups'][name]['comparisons'][label])
            all_refs[label].extend(refs)
        all_values.extend(values)
        output[name] = {**stats, 'comparisons': comparisons}
    aggregate = {label: paired(all_values, refs) for label, refs in all_refs.items()}
    for label, value in aggregate.items():
        equal_fields(value, saved['aggregate_descriptive_policy_input_pairs'][label])
    return {'cases_verified': len(records), 'compact_schedules_verified': len(records),
            'unique_inputs': len(fixtures), 'groups': output, 'aggregate_descriptive_policy_input_pairs': aggregate}


def variance(bank):
    records, fixtures = load_cases(bank, verify_variance)
    need(len(records) == 2048 and len(fixtures) == 1024, 'Variance coverage must be 2048 cases/1024 inputs')
    groups = defaultdict(list)
    for row in records.values():
        groups[row['policy'], row['task']['trial']].append(row)
    details = []
    for (policy, trial), rows in sorted(groups.items()):
        need(len(rows) == 256 and {r['task']['index'] for r in rows} == set(range(256)), 'Variance trial coverage')
        terminal = [r['terminal_normalized_objective'] for r in rows]
        for depth in range(min(len(r['actions']) for r in rows)):
            details.append({'policy': policy, 'trial': trial, 'depth': depth,
                **depth_statistics([r['trace'][depth]['partial'] for r in rows],
                                   [r['trace'][depth]['dp'] for r in rows], terminal)})
    saved_details = {(r['policy'], r['trial'], r['depth']): r for r in bank.json('depth_statistics.json')}
    need(len(details) == len(saved_details), 'Variance depth coverage')
    for row in details:
        # Identity errors are floating point diagnostics, not variance estimates.
        equal_fields({k: v for k, v in row.items() if k != 'identity_relative_error'},
                     saved_details[row['policy'], row['trial'], row['depth']])
    summary = {}
    for policy in ['learned', 'random']:
        rows = [r for r in details if r['policy'] == policy]
        ratios = [r['variance_ratio'] for r in rows if r['variance_ratio'] is not None]
        result = {'comparisons': len(rows), 'condition_holds': sum(r['condition_holds'] for r in rows),
                  'condition_fails': sum(not r['condition_holds'] for r in rows),
                  'variance_reduced': sum(r['variance_reduced'] for r in rows), 'finite_ratios': len(ratios),
                  'mean_variance_ratio': statistics.mean(ratios) if ratios else None,
                  'min_variance_ratio': min(ratios) if ratios else None}
        equal_fields(result, bank.json('summary.json')[policy])
        summary[policy] = result
    return {'cases_verified': len(records), 'summary': summary, 'depth_statistics': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--phase', choices=['all'] + PHASES, default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    functions = dict(zip(PHASES, [static, dynamic, public, variance, ablation]))
    for phase in PHASES if args.phase == 'all' else [args.phase]:
        bank = Bank(args.root, phase)
        try:
            result = functions[phase](bank)
            result['archive_sha256'] = bank.meta['sha256']
            result['status'] = 'verified'
            target = args.output / (phase + '.json')
            target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print(f'{phase}: verified, wrote {target}', flush=True)
        finally:
            bank.close()


if __name__ == '__main__':
    main()
