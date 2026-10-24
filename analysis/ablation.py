"""Five-seed curves with the manuscript's per-run smoothing and shared prefix."""
import statistics
from verification.schedules import need

SEEDS = [42, 43, 44, 45, 46]
COMPONENTS = ['component_baseline', 'component_chain_lb_reward',
              'component_no_dp_lb_diff', 'component_no_pt_node']
CURRICULA = ['cl_peach', 'cl_no_training_events', 'cl_direct']
COMPONENT_UPDATES = [1, 201, 401, 601, 800]


def calculate(bank):
    runs = {}
    for arm in COMPONENTS + CURRICULA:
        for seed in SEEDS:
            name = f'{arm}_s{seed}'
            rows = bank.jsonl(f'runs/{name}/metrics.jsonl')
            need([r['total_updates'] for r in rows] == list(range(1, len(rows) + 1)),
                 f'Noncontiguous updates: {name}')
            need({r['seed'] for r in rows} == {seed}, f'Seed mismatch: {name}')
            validation = {r['total_updates']: r['validation_objective'] for r in rows
                          if r['validation_objective'] is not None}
            if arm in COMPONENTS:
                need(len(rows) == 800 and list(validation) == COMPONENT_UPDATES,
                     f'Component protocol mismatch: {name}')
            meta = next(r for r in bank.index['runs'] if r['arm'] == name)
            need(meta['actual_total_updates'] == len(rows), f'Run budget mismatch: {name}')
            need(meta['selected_completed_updates'] in validation, f'Checkpoint selection outside validation grid: {name}')
            runs[arm, seed] = {'validation': validation, 'loss': [r['value_loss'] for r in rows]}
    budgets = {seed: min(max(runs[arm, seed]['validation']) for arm in CURRICULA) for seed in SEEDS}
    need({str(k): v for k, v in budgets.items()} == bank.index['matched_budgets'], 'Matched budgets changed')
    common = min(budgets.values())
    need(common == bank.index['common_curriculum_completed_update'] and bank.index['seeds'] == SEEDS
         and bank.index['critic_trailing_window'] == 20, 'Ablation protocol changed')
    grid = list(range(1, common + 1, 20))
    need(grid[-1] == common, 'Common budget is off the validation grid')
    curves = {}
    for arm in COMPONENTS + CURRICULA:
        updates = COMPONENT_UPDATES if arm in COMPONENTS else grid
        curves[arm] = summarize({seed: [(u, runs[arm, seed]['validation'][u]) for u in updates]
                                 for seed in SEEDS})
    for arm in COMPONENTS:
        per_seed = {}
        for seed in SEEDS:
            loss = runs[arm, seed]['loss']
            per_seed[seed] = [(i + 1, statistics.mean(loss[max(0, i - 19):i + 1]))
                              for i in range(len(loss))]
        curves[arm + '/critic'] = summarize(per_seed)
    return {'training_seeds': SEEDS, 'matched_budgets': budgets,
            'common_curriculum_updates': common, 'critic_trailing_window': 20, 'curves': curves}


def summarize(series):
    updates = [u for u, _ in series[SEEDS[0]]]
    need(all([u for u, _ in points] == updates for points in series.values()), 'Seed grids differ')
    rows = []
    for index, update in enumerate(updates):
        values = [series[seed][index][1] for seed in SEEDS]
        need(all(v > 0 for v in values), 'Log curves require positive values')
        rows.append({'update': update, 'mean': statistics.mean(values),
                     'sample_sd': statistics.stdev(values), 'per_seed': dict(zip(SEEDS, values))})
    return rows
