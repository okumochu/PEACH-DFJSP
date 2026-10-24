#!/usr/bin/env python3
"""Train static makespan SD1/SD2 policies using explicit shared training controls."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.training_settings import FIELDS, INTEGERS, BOOLEANS, FLOATS, load_settings, build_overrides, build_scale_curriculum, save_settings


def parse_mlp_dims(text):
    try:
        result = [int(x) for x in text.split(',')]
        if not result or any(x < 1 for x in result): raise ValueError()
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated positive MLP widths') from exc


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--settings', type=Path, default=Path(__file__).with_name('settings.json'))
    parser.add_argument('--size', required=True, help='Target training and validation size')
    parser.add_argument('--data_generator', choices=('SD1', 'SD2'), required=True)
    for name in FIELDS:
        if name == 'training_size':
            continue  # --size is the sole target-size name for this entry point.
        opts = {'default': argparse.SUPPRESS}
        if name in BOOLEANS: opts['choices'] = ('on', 'off')
        elif name == 'value_normalization': opts['choices'] = ('popart', 'none')
        elif name == 'stopping_rule': opts['choices'] = ('readiness_patience', 'fixed_budget')
        elif name == 'mlp_hidden_dims': opts['type'] = parse_mlp_dims
        else: opts['type'] = int if name in INTEGERS or name == 'seed' else float if name in FLOATS else str
        parser.add_argument('--' + name, **opts)
    parser.add_argument('--experiment_name')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--wandb_run_name')
    parser.add_argument('--wandb_project')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--print_config', action='store_true')
    return parser


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    explicit = {'initial_size': '10x5', 'validation_interval': 100, 'num_envs': 20,
                'scale_cl': True, 'value_normalization': 'popart', 'training_events': False,
                **{k:v for k,v in vars(args).items() if k in FIELDS},
                'training_size': args.size}
    for key in BOOLEANS:
        if isinstance(explicit.get(key), str): explicit[key] = explicit[key] == 'on'
    try:
        settings = load_settings(args.settings, explicit)
        if settings['training_events']:
            raise ValueError('Public makespan training requires training_events=off')
        overrides = build_overrides(settings)
        if not settings['scale_cl'] and settings['stopping_rule'] == 'fixed_budget':
            overrides['scale_curriculum'] = build_scale_curriculum(settings)
        # These shared dynamic-trainer options have no meaning in the static path.
        overrides.pop('scale_event_magnitudes_to_size', None)
        overrides.pop('disable_curriculum', None)
        overrides['training_events'] = False
        if args.resume:
            if not args.resume.is_file(): raise ValueError('Resume snapshot does not exist')
            overrides['resume_path'] = str(args.resume.resolve())
            overrides['result_dir'] = str(args.resume.resolve().parent)
        for key in ('wandb_run_name', 'wandb_project'):
            if getattr(args, key) is not None: overrides[key] = getattr(args, key)
        if args.disable_wandb: overrides['wandb_enabled'] = False
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps({'generator_type': args.data_generator, 'training_settings': settings,
                          'resolved_overrides': overrides}, indent=2))
        return
    from src.utils.config_util import create_hgt_single_obj_config
    from src.HGT_single_objective.hgt_trainer_so import HGTSingleObjTrainer
    config = create_hgt_single_obj_config(
        experiment_name=args.experiment_name or f'exp_so_{args.data_generator}_{args.size}',
        generator_type=args.data_generator, **overrides)
    save_settings(config.result_dir, overrides['training_settings'], config.scale_curriculum)
    outcome = HGTSingleObjTrainer(config).train()
    print(f'Static training outcome: {outcome}', flush=True)
    if outcome in ('budget_exhausted', 'diverged', 'stuck'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
