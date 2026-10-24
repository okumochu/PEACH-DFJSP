"""Public training controls and the shared readiness/patience stopping rule."""
from pathlib import Path
import json
import math
import re

from src.utils.scale_curriculum import build_auto_ladder, DESCEND, DIVERGED, STUCK

FIELDS = (
    'scale_cl', 'initial_size', 'training_size', 'stopping_rule',
    'early_stopping_patience', 'validation_interval', 'maximum_updates',
    'value_normalization', 'scale_growth_factor', 'stage_minimum_updates',
    'stage_maximum_updates', 'stage_explained_variance', 'num_envs', 'seed',
    'device', 'micro_batch_size', 'hidden_dim', 'num_heads', 'num_layers',
    'mlp_hidden_dims', 'lr', 'training_events',
)
BOOLEANS = {'scale_cl', 'training_events'}
FLOATS = {'scale_growth_factor', 'stage_explained_variance', 'lr'}
INTEGERS = {
    'early_stopping_patience', 'validation_interval', 'maximum_updates',
    'stage_minimum_updates', 'stage_maximum_updates', 'num_envs',
    'micro_batch_size', 'hidden_dim', 'num_heads', 'num_layers',
}


def validate(settings):
    missing, unknown = set(FIELDS) - settings.keys(), settings.keys() - set(FIELDS)
    if missing or unknown:
        raise ValueError(f'Settings keys: missing={sorted(missing)}, unknown={sorted(unknown)}')
    for key in BOOLEANS:
        if type(settings[key]) is not bool:
            raise ValueError(f'{key} must be true or false')
    for key in INTEGERS:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(settings['seed']) is not int or settings['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('initial_size', 'training_size'):
        if not isinstance(settings[key], str) or not re.fullmatch(r'[1-9][0-9]*x[1-9][0-9]*', settings[key]):
            raise ValueError(f'{key} must be JOBSxMACHINES, e.g. 20x10')
    if settings['stopping_rule'] not in ('readiness_patience', 'fixed_budget'):
        raise ValueError('stopping_rule must be readiness_patience or fixed_budget')
    if settings['stage_minimum_updates'] < 20:
        raise ValueError('stage_minimum_updates must be at least 20 for the critic window')
    if settings['stage_maximum_updates'] <= settings['stage_minimum_updates']:
        raise ValueError('stage_maximum_updates must exceed stage_minimum_updates')
    for key in FLOATS:
        if type(settings[key]) not in (int, float) or not math.isfinite(settings[key]):
            raise ValueError(f'{key} must be finite')
    if settings['scale_growth_factor'] <= 1 or settings['lr'] <= 0:
        raise ValueError('scale_growth_factor must exceed 1 and lr must be positive')
    if not 0 <= settings['stage_explained_variance'] <= 1:
        raise ValueError('stage_explained_variance must be between 0 and 1')
    dims = settings['mlp_hidden_dims']
    if not isinstance(dims, list) or not dims or any(type(x) is not int or x < 1 for x in dims):
        raise ValueError('mlp_hidden_dims must be a nonempty list of positive integers')
    if settings['value_normalization'] not in ('popart', 'none'):
        raise ValueError('value_normalization must be popart or none')
    first = tuple(map(int, settings['initial_size'].split('x')))
    last = tuple(map(int, settings['training_size'].split('x')))
    if settings['scale_cl']:
        if any(a > b for a, b in zip(first, last)):
            raise ValueError('initial_size cannot exceed training_size in either dimension')
        if settings['value_normalization'] != 'popart':
            raise ValueError('scale_cl=true requires value_normalization=popart')
    if not isinstance(settings['device'], str) or not settings['device']:
        raise ValueError('device must be a nonempty string')
    return settings


def load_settings(path, overrides=None):
    settings = json.loads(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError('Settings must be a JSON object')
    settings.update(overrides or {})
    return validate(settings)


def build_overrides(settings):
    validate(settings)
    target = tuple(map(int, settings['training_size'].split('x')))
    overrides = {
        'n_j': target[0], 'n_m': target[1],
        'stopping_rule': settings['stopping_rule'],
        'early_stop_patience': settings['early_stopping_patience'],
        'max_updates': settings['maximum_updates'],
        'update_per_validate': settings['validation_interval'],
        'use_popart': settings['value_normalization'] == 'popart',
        'disable_curriculum': True,
        'scale_event_magnitudes_to_size': True,
        'training_events': settings['training_events'],
        'checkpoint_every': 1,
        **{key: settings[key] for key in (
            'seed', 'device', 'num_envs', 'micro_batch_size',
            'hidden_dim', 'num_heads', 'lr', 'mlp_hidden_dims')},
        'num_hgt_layers': settings['num_layers'],
        'training_settings': dict(settings, selection='best_validation_objective'),
    }
    if overrides['use_popart']:
        overrides['popart_beta'] = 0.1
    if settings['scale_cl'] or settings['stopping_rule'] == 'readiness_patience':
        overrides['scale_curriculum'] = build_scale_curriculum(settings)
    return overrides


def build_scale_curriculum(settings):
    """Resolve the declared target or scale ladder, including all gate controls."""
    validate(settings)
    target = tuple(map(int, settings['training_size'].split('x')))
    start = tuple(map(int, settings['initial_size'].split('x'))) if settings['scale_cl'] else target
    stages = ([(target[0], target[1], settings['micro_batch_size'])] if start == target
              else build_auto_ladder(start, target, settings['scale_growth_factor'], settings['micro_batch_size']))
    return {
        'enabled': True, 'stages': stages,
        'min_stage_updates': settings['stage_minimum_updates'],
        'max_stage_updates': settings['stage_maximum_updates'],
        'ev_threshold': settings['stage_explained_variance'],
        'window': 10, 'plateau_tol': 0.5, 'cv_tol': 0.8,
        'div_factor': 5.0, 'div_patience': 10,
        'final_hold_updates': None, 'settle_updates': None, 'descend_patience': None,
    }


def require_training_controls(config, *, scale_required=False):
    """Reject incomplete run controls before allocating a model or creating files."""
    positive = (
        'num_envs', 'max_updates', 'reset_env_episode', 'early_stop_patience',
        'update_per_validate', 'validation_set_size', 'checkpoint_every',
        'micro_batch_size',
    )
    required = (*positive, 'seed', 'validation_seed', 'stopping_rule',
                'training_events', 'use_popart')
    missing = [key for key in required if not hasattr(config, key)]
    if missing:
        raise ValueError('Missing explicit training controls: ' + ', '.join(missing))
    for key in positive:
        value = getattr(config, key)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'validation_seed'):
        value = getattr(config, key)
        if type(value) is not int or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('training_events', 'use_popart'):
        if type(getattr(config, key)) is not bool:
            raise ValueError(f'{key} must be true or false')
    if config.stopping_rule not in ('readiness_patience', 'fixed_budget'):
        raise ValueError('Unknown stopping_rule')
    scale = getattr(config, 'scale_curriculum', None)
    if scale is not None and (not isinstance(scale, dict) or type(scale.get('enabled')) is not bool):
        raise ValueError('scale_curriculum requires an explicit enabled boolean')
    if (scale_required or config.stopping_rule == 'readiness_patience') and not (scale and scale['enabled']):
        raise ValueError('This training protocol requires an explicit enabled scale_curriculum')
    if scale and scale.get('enabled'):
        required_scale = (
            'stages', 'min_stage_updates', 'max_stage_updates', 'ev_threshold',
            'window', 'plateau_tol', 'cv_tol', 'div_factor', 'div_patience',
            'final_hold_updates', 'settle_updates', 'descend_patience',
        )
        missing = [key for key in required_scale if key not in scale]
        if missing:
            raise ValueError('Missing explicit scale controls: ' + ', '.join(missing))
        if not scale['stages'] or any(
                len(stage) != 3 or any(type(value) is not int or value < 1 for value in stage)
                for stage in scale['stages']):
            raise ValueError('Scale stages require explicit positive jobs, machines and microbatch size')


class ValidationStopController:
    """Count validation patience only after the target stage first becomes ready.

    Readiness is latched until a descent or failure. The validation on the
    gate-opening update is not counted. A strict validation improvement resets
    patience; checkpoint selection itself remains the trainer's responsibility.
    All update indices here are counts of completed PPO updates (one-based).
    """
    def __init__(self, rule, patience):
        if rule not in ('readiness_patience', 'fixed_budget'):
            raise ValueError(f'Unknown stopping rule: {rule}')
        if type(patience) is not int or patience < 1:
            raise ValueError('patience must be a positive integer')
        self.rule, self.patience = rule, patience
        self.ready_update = None
        self.non_improving = 0
        self.last_validation_update = None

    def observe_gate(self, total_updates, scale_controller, decision=None):
        if self.rule == 'fixed_budget':
            return
        if scale_controller is None:
            raise ValueError('Readiness stopping requires a scale controller, including for one fixed scale')
        if decision in (DESCEND, DIVERGED, STUCK) or not scale_controller.on_final_stage:
            self.ready_update = None
            self.non_improving = 0
        elif self.ready_update is None and scale_controller.stage_ready():
            self.ready_update = int(total_updates)
            self.non_improving = 0

    def observe_validation(self, total_updates, objective, improved):
        if not math.isfinite(float(objective)):
            raise ValueError('Non-finite validation objective')
        total_updates = int(total_updates)
        if self.last_validation_update is not None and total_updates <= self.last_validation_update:
            raise ValueError('Validation updates must increase strictly')
        self.last_validation_update = total_updates
        if self.rule == 'fixed_budget' or self.ready_update is None or total_updates <= self.ready_update:
            self.non_improving = 0
            return False
        self.non_improving = 0 if improved else self.non_improving + 1
        return self.non_improving >= self.patience

    def state_dict(self):
        return dict(schema=1, rule=self.rule, patience=self.patience,
                    ready_update=self.ready_update, non_improving=self.non_improving,
                    last_validation_update=self.last_validation_update)

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('schema') != 1:
            raise ValueError('Missing or unsupported stopping state')
        if state.get('rule') != self.rule or state.get('patience') != self.patience:
            raise ValueError('Resume changes stopping rule or patience')
        for key in ('ready_update', 'last_validation_update'):
            value = state.get(key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'Invalid stopping state: {key}')
        count = state.get('non_improving')
        if type(count) is not int or not 0 <= count <= self.patience:
            raise ValueError('Invalid non-improving validation count')
        if state['ready_update'] is None and count:
            raise ValueError('Patience cannot be spent before readiness')
        self.ready_update = state['ready_update']
        self.non_improving = count
        self.last_validation_update = state['last_validation_update']


def save_settings(directory, settings, resolved_stages):
    """Persist immutable scientific settings; device may change for recovery."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    record = json.loads(json.dumps(dict(settings, resolved_scale_stages=resolved_stages)))
    path = root / 'training-settings.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if {k: v for k, v in previous.items() if k != 'device'} != {k: v for k, v in record.items() if k != 'device'}:
            raise ValueError('Resume changes recorded scientific training settings')
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)
    return record
