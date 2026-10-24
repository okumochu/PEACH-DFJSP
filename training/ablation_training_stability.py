#!/usr/bin/env python3
"""Train one reward/representation ablation at the shared 800-update budget.

All four arms train directly at 20x10 with target dynamic events, no DDT or
scale curriculum, and no validation early stopping. Historical Block A used
raw value targets; this component protocol therefore keeps PopArt disabled.
Architecture and learning rate resolve through the shared configuration.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.training_settings import build_overrides, load_settings, save_settings

ABLATION_CHOICES = ('baseline', 'chain_lb_reward', 'no_dp_lb_diff', 'no_pt_node')
FIXED_SETTINGS = {
    'stopping_rule': 'fixed_budget',
    'maximum_updates': 800,
    'initial_size': '20x10',
    'training_size': '20x10',
    'scale_cl': False,
    'training_events': True,
    'value_normalization': 'none',
    'num_envs': 20,
    'validation_interval': 200,
}


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--ablation', required=True, choices=ABLATION_CHOICES)
    parser.add_argument('--settings', type=Path, default=Path(__file__).with_name('settings.json'),
                        help='Shared settings file; component scientific conditions are fixed by this launcher')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44, 45, 46), default=42)
    parser.add_argument('--device', help='Training device, e.g. cuda:0 or cpu')
    parser.add_argument('--micro_batch_size', type=int, help='Operational graph chunk size; effective minibatch stays shared')
    parser.add_argument('--checkpoint_every', type=int, default=1,
                        help='Committed updates between full resume snapshots (default: 1)')
    parser.add_argument('--experiment_name', help='Default: ablation_<arm>_s<seed>')
    parser.add_argument('--result_dir', type=Path, help='Explicit new-run output directory')
    parser.add_argument('--resume', type=Path, help='Resume full state in the snapshot directory under the shared contract')
    parser.add_argument('--wandb_run_name')
    parser.add_argument('--wandb_project')
    parser.add_argument('--disable_wandb', action='store_true')
    parser.add_argument('--print_config', '--print-config', action='store_true',
                        help='Print the complete resolved configuration without training or filesystem writes')
    return parser


def resolve_settings(args):
    explicit = dict(FIXED_SETTINGS, seed=args.seed)
    for key in ('device', 'micro_batch_size'):
        value = getattr(args, key)
        if value is not None:
            explicit[key] = value
    settings = load_settings(args.settings, explicit)
    if args.checkpoint_every < 1:
        raise ValueError('--checkpoint_every must be a positive integer')
    overrides = build_overrides(settings)
    overrides.update(
        validation_seed=42,
        validation_set_size=30,
        checkpoint_every=args.checkpoint_every,
        use_pt_nodes=args.ablation != 'no_pt_node',
        dp_lb_diff={'enabled': args.ablation != 'no_dp_lb_diff',
                    'measure': 'chain' if args.ablation == 'chain_lb_reward' else 'dp'},
        component_ablation=args.ablation,
    )
    overrides['training_settings']['component_ablation'] = args.ablation
    if args.resume is not None:
        if not args.resume.is_file():
            raise ValueError(f'--resume file does not exist: {args.resume}')
        resume = args.resume.resolve()
        if args.result_dir is not None and args.result_dir.resolve() != resume.parent:
            raise ValueError('--result_dir must equal the resume snapshot directory')
        overrides.update(resume_path=str(resume), result_dir=str(resume.parent))
    elif args.result_dir is not None:
        overrides['result_dir'] = str(args.result_dir.resolve())
    name = args.experiment_name or f'ablation_{args.ablation}_s{args.seed}'
    overrides['wandb_run_name'] = args.wandb_run_name or name
    if args.wandb_project is not None:
        overrides['wandb_project'] = args.wandb_project
    if args.disable_wandb:
        overrides['wandb_enabled'] = False
    return name, overrides


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        name, overrides = resolve_settings(args)
        from src.utils.config_util import create_hgt_config
        config = create_hgt_config(name, create_result_dir=not args.print_config, **overrides)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps({'ablation': args.ablation,
                          'training_settings': overrides['training_settings'],
                          'resolved_config': vars(config)}, indent=2, default=str))
        return
    record = save_settings(config.result_dir, overrides['training_settings'],
                           getattr(config, 'scale_curriculum', None))
    print('Effective training settings:\n'+json.dumps(record, indent=2), flush=True)
    from src.HGT.hgt_trainer import HGTTrainer
    HGTTrainer(config).train()


if __name__ == '__main__':
    main()
