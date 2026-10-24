"""Render ablation curves recomputed by analysis.reproduce, without training."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.ablation import COMPONENTS, CURRICULA, SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Recomputed ablations.json')
    parser.add_argument('--output', type=Path, required=True, help='PDF, SVG or PNG figure')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    labels = ['PEACH', 'Job-chain reward', 'Partial-objective reward', 'Without PT nodes']
    cl_labels = ['PEACH', 'w/o event', 'w/o scale CL']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, names, names_labels, critic in [(axes[0], COMPONENTS, labels, False),
                                             (axes[1], COMPONENTS, labels, True),
                                             (axes[2], CURRICULA, cl_labels, False)]:
        for i, (arm, label) in enumerate(zip(names, names_labels)):
            points = data['curves'][arm + ('/critic' if critic else '')]
            x = [p['update'] for p in points]
            for seed in SEEDS:
                axis.plot(x, [p['per_seed'][str(seed)] for p in points], color=colors[i],
                          linewidth=.5, alpha=.3)
            axis.plot(x, [p['mean'] for p in points], color=colors[i], linestyle=styles[i],
                      marker=None if critic else markers[i], markersize=3, linewidth=1.2, label=label)
        axis.set_yscale('log')
        axis.set_xlabel('PPO updates')
        axis.grid(axis='y', linewidth=.4, alpha=.3)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(labelsize=9)
        axis.legend(fontsize=9, frameon=False)
    axes[0].set_ylabel('Validation objective')
    axes[1].set_ylabel('Value loss')
    axes[2].set_ylabel('Validation objective')
    captions = ['(a) State and reward', '(b) Critic loss', '(c) Event-aware scale CL']
    for axis, caption in zip(axes, captions):
        axis.text(.5, -.34, caption, transform=axis.transAxes, ha='center', va='top', fontsize=10)
    fig.subplots_adjust(left=.07, right=.99, top=.96, bottom=.28, wspace=.38)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
