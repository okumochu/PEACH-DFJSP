#!/usr/bin/env python3
"""
Gantt Chart Utility Module for FJSP Visualization

Provides reusable Gantt chart drawing functions for:
- Static and dynamic FJSP schedules
- Due date indicators
- Dynamic event markers (job arrivals, priority changes)
- Breakdown/repair period visualization
- Fixed operation highlighting (for rescheduling comparisons)

Usage:
    from src.utils.gantt_util import draw_gantt_chart, extract_schedule_from_env
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from typing import Dict, Any, List, Optional


# ============================================================================
# Color Constants for Dynamic Events (distinct from job colors)
# ============================================================================
EVENT_COLORS = {
    'job_arrival': '#D32F2F',      # Red for arrivals
    'priority_change': '#2E7D32',  # Green for priority changes
    'breakdown_period': '#BDBDBD', # Light gray for breakdown/repair areas
    'breakdown_edge': '#616161',   # Darker gray for breakdown edges
    'breakdown_line': '#B71C1C',   # Dark red for breakdown-wave vertical markers
    'combined_event': '#7B1FA2',   # Purple for coincident arrival+priority markers
}

# Border drawn around the single most-urgent reprioritized job (weight=5) after
# a priority-change event — see ``highlight_job_ids`` in draw_gantt_chart.
HIGHLIGHT_BORDER_COLOR = '#FFD400'  # vivid yellow/gold


def extract_schedule_from_env(env, env_idx: int = 0) -> Dict[str, Any]:
    """
    Extract schedule data from an FJSPEnvHGT environment for Gantt chart visualization.
    
    Args:
        env: FJSPEnvHGT environment after episode completion
        env_idx: Environment index (for batched environments)
        
    Returns:
        Dictionary containing:
        - schedule: List of operation dicts
        - job_data: List of job info dicts
        - num_jobs, num_machines, makespan
        - breakdown_periods, interrupted_ops
    """
    # Per-env ACTIVE counts, not number_of_jobs / number_of_ops: those are
    # frozen at their initial values for the whole episode while an inline job
    # arrival grows n_active_* into the pre-allocated slots. Filtering by the
    # frozen counts silently drops every arrived job's operations from the
    # extracted schedule -- and this function feeds freeze_segment_ops, so a
    # dropped op would also vanish from the frozen prefix. Identical to the
    # frozen counts on every live path (both training and InferenceEnv rebuild
    # each segment from a static dict already grown to include arrived jobs, so
    # n_active_* == number_of_*).
    num_machines = env.number_of_machines
    _act_j = getattr(env, 'n_active_jobs', None)
    _act_o = getattr(env, 'n_active_ops', None)
    num_jobs = int(_act_j[env_idx]) if _act_j is not None else env.number_of_jobs
    num_ops = int(_act_o[env_idx]) if _act_o is not None else env.number_of_ops
    
    # Extract operation schedule from machine queues
    schedule = []
    
    for m_idx in range(num_machines):
        queue_len = env.mch_queue_len[env_idx, m_idx]
        for q_pos in range(queue_len):
            op_idx = env.mch_queue[env_idx, m_idx, q_pos]
            if op_idx >= 0 and op_idx < num_ops:
                # Find which job this operation belongs to
                job_idx = -1
                for j in range(num_jobs):
                    first_op = env.job_first_op_id[env_idx, j]
                    last_op = env.job_last_op_id[env_idx, j]
                    if first_op <= op_idx <= last_op:
                        job_idx = j
                        break
                
                if job_idx >= 0:
                    ct = float(env.unnorm_op_ct[env_idx, op_idx])
                    # Use tracked start time when available (dynamic env),
                    # fall back to ct - pt for static env
                    if hasattr(env, 'unnorm_op_start_time') and env.unnorm_op_start_time is not None:
                        start_time = float(env.unnorm_op_start_time[env_idx, op_idx])
                    else:
                        start_time = ct - float(env.op_pt[env_idx, op_idx, m_idx])
                    pt = ct - start_time

                    schedule.append({
                        'op_id': int(op_idx),
                        'job_id': int(job_idx),
                        'machine_id': int(m_idx),
                        'start_time': start_time,
                        'end_time': ct,
                        'processing_time': pt,
                    })
    
    # Extract job data
    job_data = []
    for j in range(num_jobs):
        job_data.append({
            'job_idx': j,
            'weight': int(env.job_weight[env_idx, j]),
            'due_date': float(env.job_due_date[env_idx, j]),
            'release_time': float(env.job_release_time[env_idx, j]),
            'first_op': int(env.job_first_op_id[env_idx, j]),
            'last_op': int(env.job_last_op_id[env_idx, j]),
        })
    
    # Extract breakdown periods
    breakdown_periods = getattr(env, 'breakdown_periods', [])
    breakdown_periods = [bp for bp in breakdown_periods if bp.get('env_idx', 0) == env_idx]
    
    # Extract interrupted operations
    interrupted_ops = getattr(env, 'interrupted_ops', [])
    interrupted_ops = [io for io in interrupted_ops if io.get('env_idx', 0) == env_idx]
    
    return {
        'schedule': schedule,
        'job_data': job_data,
        'num_jobs': num_jobs,
        'num_machines': num_machines,
        'makespan': float(env.current_makespan[env_idx]),
        'twt': float(env.current_twt[env_idx]),
        'breakdown_periods': breakdown_periods,
        'interrupted_ops': interrupted_ops,
    }


def draw_gantt_chart(
    schedule: List[Dict],
    num_machines: int,
    num_jobs: int,
    output_path: str,
    title: str = 'FJSP Schedule',
    job_data: Optional[List[Dict]] = None,
    dynamic_events: Optional[Dict[str, Any]] = None,
    breakdown_periods: Optional[List[Dict]] = None,
    interrupted_ops: Optional[List[Dict]] = None,
    fixed_op_ids: Optional[set] = None,
    highlight_job_ids: Optional[set] = None,
    show_due_dates: bool = True,
    figsize: Optional[tuple] = None,
    dpi: int = 150,
):
    """
    Draw a Gantt chart for FJSP schedules with optional dynamic event markers.
    
    Features:
    - Operation bars colored by job with labels
    - Due date vertical dashed lines (per job)
    - Breakdown periods as gray hatched rectangles
    - Interrupted operations with partial completion
    - Dynamic event markers (job arrivals, priority changes)
    - Fixed operation highlighting for rescheduling
    
    Args:
        schedule: List of operation dicts with keys:
            job_id, machine_id, start_time, end_time, (optional: op_id, processing_time)
        num_machines: Number of machines
        num_jobs: Number of jobs (for color palette)
        output_path: Path to save the figure
        title: Chart title
        job_data: Optional list of job info dicts with keys:
            job_idx, weight, due_date, release_time, first_op, last_op
        dynamic_events: Optional dict with event times:
            new_job_arrival: list of float times
            priority_change: list of float times
            machine_breakdown: list of {'time': float, ...} (used for annotation only)
        breakdown_periods: Optional list of breakdown period dicts:
            machine_idx, start_time, end_time
        interrupted_ops: Optional list of interrupted operation dicts:
            machine_idx, op_idx, start_time, interrupt_time
        fixed_op_ids: Optional set of fixed operation IDs (shown with different style)
        highlight_job_ids: Optional set of job IDs whose operation bars are
            outlined in yellow (HIGHLIGHT_BORDER_COLOR). Used to flag the
            most-urgent reprioritized job after a priority-change event.
        show_due_dates: Whether to show due date vertical lines
        figsize: Figure size tuple
        dpi: Output DPI
    """
    # Density flag: only suppresses per-job legend entries and due-date dashes
    # on large instances (both would drown the canvas). Bars use tab20 +
    # black-border + J# regardless — same style as the GIF renderer.
    is_dense = (num_jobs * num_machines) >= 500

    colors = plt.cm.tab20(np.linspace(0, 1, max(num_jobs, 20)))

    # Determine initial job count (jobs with release_time == 0 are initial)
    if job_data:
        initial_num_jobs = sum(1 for jd in job_data if jd.get('release_time', 0) == 0)
    else:
        initial_num_jobs = num_jobs

    # Auto-size when caller didn't pin figsize: scale with problem dimensions so
    # bars remain visible at 100x20 without user-side tuning.
    if figsize is None:
        figsize = (
            max(14.0, num_jobs * 0.12),
            max(6.0, num_machines * 0.35),
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    bar_height = 0.6

    # Find makespan
    if schedule:
        makespan = max(op['end_time'] for op in schedule)
    else:
        makespan = 100  # Default if empty
    
    # ========================================
    # 1. Draw breakdown periods (background)
    # ========================================
    if breakdown_periods:
        for bp in breakdown_periods:
            machine_idx = bp['machine_idx']
            start_time = bp['start_time']
            end_time = bp['end_time']
            duration = end_time - start_time
            
            # Draw hatched rectangle for breakdown/repair period
            rect = Rectangle(
                (start_time, machine_idx - bar_height/2),
                min(duration, makespan * 1.5 - start_time),
                bar_height,
                facecolor=EVENT_COLORS['breakdown_period'],
                edgecolor=EVENT_COLORS['breakdown_edge'],
                linewidth=1,
                hatch='///',
                alpha=0.8,
                zorder=1
            )
            ax.add_patch(rect)
            
            # Add "Repair" label for wider bars
            if duration > 5:
                ax.text(
                    start_time + min(duration, makespan - start_time) / 2,
                    machine_idx,
                    'Repair',
                    ha='center',
                    va='center',
                    fontsize=7,
                    fontweight='bold',
                    color='gray',
                    zorder=2
                )
    
    # ========================================
    # 2. Draw interrupted operations (partial execution)
    # ========================================
    if interrupted_ops and job_data:
        for io in interrupted_ops:
            machine_idx = io['machine_idx']
            start_time = io['start_time']
            interrupt_time = io['interrupt_time']
            op_idx = io['op_idx']
            
            # Find job for this operation
            job_idx = -1
            for job in job_data:
                first_op = job.get('first_op', 0)
                last_op = job.get('last_op', 0)
                if first_op <= op_idx <= last_op:
                    job_idx = job['job_idx']
                    break
            
            if job_idx >= 0:
                partial_duration = interrupt_time - start_time
                rect = Rectangle(
                    (start_time, machine_idx - bar_height/2),
                    partial_duration,
                    bar_height,
                    facecolor=colors[job_idx % len(colors)],
                    edgecolor='red',
                    linewidth=2,
                    linestyle='--',
                    alpha=0.5,
                    zorder=2
                )
                ax.add_patch(rect)
    
    # ========================================
    # 3. Draw operation bars
    # ========================================
    for op in schedule:
        job_idx = op.get('job_id', op.get('job_idx', 0))
        machine_idx = op.get('machine_id', op.get('machine_idx', 0))
        start = op['start_time']
        duration = op['end_time'] - op['start_time']
        
        # Border style precedence: urgent (yellow) > fixed (blue) > normal.
        # Urgent bars sit at a higher zorder so the yellow outline is never
        # occluded by neighbouring bars.
        is_fixed = fixed_op_ids is not None and op.get('op_id') in fixed_op_ids
        is_urgent = highlight_job_ids is not None and job_idx in highlight_job_ids

        if is_urgent:
            edge, lw, z = HIGHLIGHT_BORDER_COLOR, 2.8, 5
        elif is_fixed:
            edge, lw, z = 'darkblue', 2, 3
        else:
            edge, lw, z = 'black', 1, 3

        rect = Rectangle(
            (start, machine_idx - bar_height/2),
            duration,
            bar_height,
            facecolor=colors[job_idx % len(colors)],
            edgecolor=edge,
            linewidth=lw,
            zorder=z
        )
        ax.add_patch(rect)

        if duration > 1:
            ax.text(
                start + duration/2,
                machine_idx,
                f'J{job_idx}',
                ha='center',
                va='center',
                fontsize=8,
                fontweight='bold',
                color='white',
                zorder=z + 1
            )
    
    # ========================================
    # 4. Draw due date vertical lines
    # ========================================
    # Skip on dense instances — 100 colored dashed lines bury the schedule.
    if show_due_dates and job_data and not is_dense:
        for job in job_data:
            job_idx = job['job_idx']
            due_date = job.get('due_date', np.inf)
            if due_date < np.inf and due_date <= makespan * 1.5:
                ax.axvline(
                    x=due_date,
                    color=colors[job_idx % len(colors)],
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=5
                )
    
    # ========================================
    # 5. Draw dynamic event markers
    # ========================================
    if dynamic_events:
        # Job arrivals use dashed black markers so the event remains visible
        # in grayscale print.
        for i, arrival_event in enumerate(dynamic_events.get('new_job_arrival', [])):
            if isinstance(arrival_event, dict):
                arrival_time = arrival_event.get('time', 0)
                label = arrival_event.get('label', f'Arrival {i + 1}')
            else:
                arrival_time = arrival_event
                label = f'Arrival {i + 1}'
            
            if arrival_time <= makespan * 1.2:
                ax.axvline(
                    x=arrival_time,
                    color='black',
                    linestyle='--',
                    linewidth=2.1,
                    alpha=0.9,
                    zorder=5
                )
                if is_dense:
                    ax.text(
                        arrival_time,
                        num_machines - 0.25,
                        label,
                        rotation=90,
                        ha='right',
                        va='top',
                        fontsize=7,
                        color='black',
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.75, pad=1.0),
                        zorder=6,
                    )
        
        # Priority shocks use solid black markers and include the affected-job
        # count when the event record provides it.
        for i, priority_event in enumerate(dynamic_events.get('priority_change', [])):
            if isinstance(priority_event, dict):
                priority_time = priority_event.get('time', 0)
                affected = priority_event.get('affected_count')
                if affected is None:
                    affected = len(priority_event.get('job_indices') or [])
                label = priority_event.get('label')
                if label is None:
                    label = f'Shock {i + 1}'
                    if affected:
                        label = f'{label}: {affected} jobs'
            else:
                priority_time = priority_event
                label = f'Shock {i + 1}'
            
            if priority_time <= makespan * 1.2:
                ax.axvline(
                    x=priority_time,
                    color='black',
                    linestyle='-',
                    linewidth=2.8,
                    alpha=0.9,
                    zorder=5
                )
                if is_dense:
                    ax.text(
                        priority_time,
                        num_machines - 0.25,
                        label,
                        rotation=90,
                        ha='left',
                        va='top',
                        fontsize=7,
                        fontweight='bold',
                        color='black',
                        bbox=dict(facecolor='white', edgecolor='black',
                                  linewidth=0.4, alpha=0.85, pad=1.0),
                        zorder=6,
                    )

        # Coincident arrival + priority change (coupled demand shock): a single
        # purple solid marker instead of overlapping dashed+solid black lines.
        # The caller is responsible for placing only same-time pairs here (and
        # keeping lone arrivals/priorities in their own keys).
        for i, combo_event in enumerate(dynamic_events.get('combined_event', [])):
            if isinstance(combo_event, dict):
                combo_time = combo_event.get('time', 0)
                label = combo_event.get('label', 'Arrival + Priority')
            else:
                combo_time = combo_event
                label = 'Arrival + Priority'

            if combo_time <= makespan * 1.2:
                ax.axvline(
                    x=combo_time,
                    color=EVENT_COLORS['combined_event'],
                    linestyle='-',
                    linewidth=2.8,
                    alpha=0.9,
                    zorder=5
                )
                if is_dense:
                    ax.text(
                        combo_time,
                        num_machines - 0.25,
                        label,
                        rotation=90,
                        ha='left',
                        va='top',
                        fontsize=7,
                        fontweight='bold',
                        color=EVENT_COLORS['combined_event'],
                        bbox=dict(facecolor='white',
                                  edgecolor=EVENT_COLORS['combined_event'],
                                  linewidth=0.4, alpha=0.85, pad=1.0),
                        zorder=6,
                    )

        # Machine breakdowns use dash-dot dark-red markers (distinct from the
        # solid priority / dashed arrival lines). Entries are grouped by time so
        # one line is drawn per wave; affected machines also appear as gray
        # repair rectangles (see breakdown_periods).
        breakdown_marks: Dict[float, int] = {}
        breakdown_labels: Dict[float, Any] = {}
        for bd in dynamic_events.get('machine_breakdown', []):
            if isinstance(bd, dict):
                bd_time = float(bd.get('time', 0))
                breakdown_labels.setdefault(bd_time, bd.get('label'))
            else:
                bd_time = float(bd)
                breakdown_labels.setdefault(bd_time, None)
            breakdown_marks[bd_time] = breakdown_marks.get(bd_time, 0) + 1
        for w, bd_time in enumerate(sorted(breakdown_marks)):
            label = breakdown_labels.get(bd_time)
            if label is None:
                n = breakdown_marks[bd_time]
                label = f'Breakdown {w + 1}'
                if n > 1:
                    label = f'{label}: {n} machines'
            if bd_time <= makespan * 1.2:
                ax.axvline(
                    x=bd_time,
                    color=EVENT_COLORS['breakdown_line'],
                    linestyle='-.',
                    linewidth=2.4,
                    alpha=0.9,
                    zorder=5
                )
                if is_dense:
                    ax.text(
                        bd_time,
                        num_machines - 0.25,
                        label,
                        rotation=90,
                        ha='right',
                        va='top',
                        fontsize=7,
                        fontweight='bold',
                        color=EVENT_COLORS['breakdown_line'],
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.8, pad=1.0),
                        zorder=6,
                    )

    # ========================================
    # 6. Configure axes
    # ========================================
    ax.set_xlim(-1, makespan * 1.1)
    ax.set_ylim(-0.5, num_machines + 0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Machines', fontsize=12)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(False)
    
    # ========================================
    # 7. Create legend
    # ========================================
    legend_elements = []

    # Dense instances: skip per-job legend entries (they would occupy the right
    # side of the canvas and crowd out the schedule). Event-marker entries below
    # still appear so breakdown / arrival / priority annotations remain legible.
    if not is_dense:
        # Job colors - show all jobs with their weights
        new_job_counter = 0
        for j in range(num_jobs):
            weight_str = ''
            if job_data and j < len(job_data):
                weight_str = f' (w={job_data[j].get("weight", "?")})'
            # Mark dynamically arrived jobs (release_time > 0) with "New Job" prefix
            is_new_job = job_data and j < len(job_data) and job_data[j].get('release_time', 0) > 0
            if is_new_job:
                new_job_counter += 1
                label = f'New Job {new_job_counter}{weight_str}'
            else:
                label = f'Job {j}{weight_str}'
            patch = mpatches.Patch(
                color=colors[j],
                label=label
            )
            legend_elements.append(patch)

    # Breakdown/Repair
    if breakdown_periods:
        legend_elements.append(mpatches.Patch(
            facecolor=EVENT_COLORS['breakdown_period'],
            edgecolor=EVENT_COLORS['breakdown_edge'],
            hatch='///',
            label='Breakdown/Repair'
        ))
    
    # Dynamic event markers - legend uses gray since actual lines use job-specific colors
    if dynamic_events:
        if dynamic_events.get('new_job_arrival'):
            legend_elements.append(plt.Line2D(
                [0], [0],
                color='black',
                linestyle='--',
                linewidth=2.1,
                label='Rush Arrival Wave'
            ))
        if dynamic_events.get('priority_change'):
            legend_elements.append(plt.Line2D(
                [0], [0],
                color='black',
                linestyle='-',
                linewidth=2.8,
                label='Priority Shock'
            ))
        if dynamic_events.get('combined_event'):
            legend_elements.append(plt.Line2D(
                [0], [0],
                color=EVENT_COLORS['combined_event'],
                linestyle='-',
                linewidth=2.8,
                label='Arrival + Priority Shock'
            ))
        if dynamic_events.get('machine_breakdown'):
            legend_elements.append(plt.Line2D(
                [0], [0],
                color=EVENT_COLORS['breakdown_line'],
                linestyle='-.',
                linewidth=2.4,
                label='Breakdown Wave'
            ))

    # Fixed operations
    if fixed_op_ids:
        legend_elements.append(mpatches.Patch(
            facecolor='white',
            edgecolor='darkblue',
            linewidth=2,
            label='Fixed Op'
        ))

    # Most-urgent reprioritized job (weight=5) after a priority shock
    if highlight_job_ids:
        legend_elements.append(mpatches.Patch(
            facecolor='white',
            edgecolor=HIGHLIGHT_BORDER_COLOR,
            linewidth=2.8,
            label='Most-urgent job (w=5)'
        ))

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=8,
            ncol=2
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    


def _build_op_to_job_map(job_data: Optional[List[Dict]]) -> Dict[int, int]:
    """Build a {op_idx: job_idx} dict from ``job_data`` so callers can look up a
    job in O(1) instead of linearly scanning ``first_op``..``last_op`` ranges."""
    out: Dict[int, int] = {}
    if not job_data:
        return out
    for job in job_data:
        first = int(job.get('first_op', -1))
        last = int(job.get('last_op', -2))
        job_idx = int(job['job_idx'])
        for op_idx in range(first, last + 1):
            out[op_idx] = job_idx
    return out


def _describe_event_set(event_set: Optional[List[Dict[str, Any]]],
                          time_now: float) -> str:
    """One-line description for the title of a GIF frame."""
    if not event_set:
        return 'Initial plan'
    parts = []
    for e in event_set:
        et = e.get('type')
        if et == 'job_arrival':
            parts.append('Job arrives')
        elif et == 'machine_breakdown':
            parts.append(f"M{e.get('machine_idx', '?')} breaks down")
        elif et == 'priority_change':
            parts.append('Priorities flipped')
        else:
            parts.append(str(et))
    return f"{' + '.join(parts)} @ t={time_now:.1f}"


def _draw_snapshot_frame(ax, snapshot: Dict[str, Any],
                          colors, num_machines: int,
                          x_max: float, title_prefix: str) -> None:
    """Render one snapshot onto a matplotlib Axes with fixed axis bounds.

    Expected snapshot schema (produced by
    ``InferenceEnv.run(..., collect_snapshots=True)``):
        - ``schedule``: list of op dicts (``op_id``, ``job_id``, ``machine_id``,
          ``start_time``, ``end_time``, ``processing_time``)
        - ``job_data``: list of job dicts (``job_idx``, ``weight``, ``due_date``,
          ``release_time``, ``first_op``, ``last_op``)
        - ``fixed_op_ids``: set of op_ids frozen before this segment
        - ``breakdown_periods``, ``interrupted_ops``: accumulated event logs
        - ``triggering_event_set``: event(s) that led to this segment, or None
        - ``time_now``: event time that triggered this segment (0.0 for seg 0)
        - ``segment_idx``, ``segment_makespan``
    """
    schedule = snapshot['schedule']
    job_data = snapshot.get('job_data', []) or []
    fixed_op_ids = snapshot.get('fixed_op_ids', set()) or set()
    breakdown_periods = snapshot.get('breakdown_periods', []) or []
    interrupted_ops = snapshot.get('interrupted_ops', []) or []
    triggering_event_set = snapshot.get('triggering_event_set')
    time_now = float(snapshot.get('time_now', 0.0))
    segment_idx = int(snapshot.get('segment_idx', 0))
    segment_makespan = float(snapshot.get('segment_makespan', 0.0))

    bar_height = 0.6
    op_to_job = _build_op_to_job_map(job_data)

    for bp in breakdown_periods:
        m = int(bp['machine_idx'])
        s = float(bp['start_time'])
        e = float(bp.get('end_time', np.inf))
        if not np.isfinite(e):
            e = x_max
        e = min(e, x_max)
        if e <= s:
            continue
        rect = Rectangle(
            (s, m - bar_height / 2),
            e - s, bar_height,
            facecolor=EVENT_COLORS['breakdown_period'],
            edgecolor=EVENT_COLORS['breakdown_edge'],
            linewidth=1, hatch='///', alpha=0.8, zorder=1,
        )
        ax.add_patch(rect)

    for io_op in interrupted_ops:
        m = int(io_op['machine_idx'])
        s = float(io_op['start_time'])
        it = float(io_op['interrupt_time'])
        op_idx = int(io_op['op_idx'])
        job_idx = op_to_job.get(op_idx)
        if job_idx is None:
            continue
        rect = Rectangle(
            (s, m - bar_height / 2),
            max(0.0, it - s), bar_height,
            facecolor=colors[job_idx % len(colors)],
            edgecolor='red', linewidth=2, linestyle='--',
            alpha=0.5, zorder=2,
        )
        ax.add_patch(rect)

    label_min_width = x_max * 0.015
    for op in schedule:
        job_idx = int(op.get('job_id', op.get('job_idx', 0)))
        m = int(op.get('machine_id', op.get('machine_idx', 0)))
        s = float(op['start_time'])
        duration = float(op['end_time']) - s
        is_fixed = op.get('op_id') in fixed_op_ids
        rect = Rectangle(
            (s, m - bar_height / 2),
            duration, bar_height,
            facecolor=colors[job_idx % len(colors)],
            edgecolor='darkblue' if is_fixed else 'black',
            linewidth=2 if is_fixed else 1,
            zorder=3,
        )
        ax.add_patch(rect)
        if duration > label_min_width:
            ax.text(
                s + duration / 2, m,
                f'J{job_idx}',
                ha='center', va='center',
                fontsize=7, fontweight='bold',
                color='white', zorder=4,
            )

    for job in job_data:
        dd = float(job.get('due_date', np.inf))
        if np.isfinite(dd) and dd <= x_max:
            ax.axvline(
                x=dd,
                color=colors[int(job['job_idx']) % len(colors)],
                linestyle='--', linewidth=1.0, alpha=0.5, zorder=5,
            )

    # Frame 0 has no triggering event — skip the current-time marker instead of
    # comparing time_now to 0 (an event firing at t=0 would otherwise be hidden).
    if triggering_event_set is not None:
        ax.axvline(x=time_now, color='red', linestyle=':',
                   linewidth=2.5, alpha=0.9, zorder=6)
        ax.text(time_now, num_machines - 0.4,
                f' t={time_now:.1f}',
                color='red', fontsize=9, fontweight='bold', zorder=6)

    ax.set_xlim(-1, x_max)
    ax.set_ylim(-0.6, num_machines - 0.4)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Machines', fontsize=11)
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.grid(False)

    evt_desc = _describe_event_set(triggering_event_set, time_now)
    ax.set_title(
        f'{title_prefix} — Segment {segment_idx}: {evt_desc}  '
        f'(segment makespan: {segment_makespan:.1f})',
        fontsize=12, fontweight='bold',
    )

    legend_elements = [
        mpatches.Patch(facecolor='lightgray', edgecolor='darkblue',
                       linewidth=2, label='Fixed op'),
        mpatches.Patch(facecolor='lightgray', edgecolor='black',
                       linewidth=1, label='Re-planned op'),
    ]
    if breakdown_periods:
        legend_elements.append(mpatches.Patch(
            facecolor=EVENT_COLORS['breakdown_period'],
            edgecolor=EVENT_COLORS['breakdown_edge'],
            hatch='///', label='Breakdown'))
    if interrupted_ops:
        legend_elements.append(mpatches.Patch(
            facecolor='lightgray', edgecolor='red',
            linewidth=2, label='Interrupted'))
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=8, ncol=1, framealpha=0.9)


def create_gantt_gif(
    snapshots: List[Dict[str, Any]],
    num_machines: int,
    total_num_jobs: int,
    output_path: str,
    title_prefix: str = 'FJSP Schedule',
    frame_duration_ms: int = 1500,
    figsize: tuple = (14, 6),
    dpi: int = 90,
) -> None:
    """Render per-segment snapshots from ``InferenceEnv`` into an animated GIF.

    Axis bounds (x: 0..max_makespan+buffer, y: num_machines) and the job color
    palette are fixed across frames so the layout never resizes mid-GIF. Fixed
    ops are highlighted with a dark-blue border; re-planned ops get the default
    black border. A red dotted vertical line marks the event time for each
    non-initial frame, and breakdown periods / interrupted ops accumulate.

    Args:
        snapshots: list returned by ``InferenceEnv.run(..., collect_snapshots=True)``.
        num_machines: y-axis extent (fixed across frames).
        total_num_jobs: number of jobs in the final segment — controls the color
            palette size so job colors are stable across frames.
        output_path: where to write the ``.gif`` file.
        title_prefix: prefix prepended to each frame's title.
        frame_duration_ms: per-frame duration.
        figsize, dpi: matplotlib figure configuration.
    """
    from PIL import Image

    if not snapshots:
        raise ValueError('create_gantt_gif: snapshots list is empty')

    max_time = 0.0
    for s in snapshots:
        for op in s['schedule']:
            if op['end_time'] > max_time:
                max_time = float(op['end_time'])
        for bp in s.get('breakdown_periods', []) or []:
            et = float(bp.get('end_time', 0.0))
            if np.isfinite(et) and et > max_time:
                max_time = et
        for job in s.get('job_data', []) or []:
            dd = float(job.get('due_date', np.inf))
            if np.isfinite(dd) and dd > max_time:
                max_time = dd
        if float(s.get('time_now', 0.0)) > max_time:
            max_time = float(s['time_now'])
    x_max = max_time * 1.08 + 1.0

    colors = plt.cm.tab20(np.linspace(0, 1, max(total_num_jobs, 20)))

    # Reuse one figure across frames and skip the PNG encode/decode roundtrip
    # by reading the canvas buffer directly — this is the dominant cost for
    # many-frame GIFs.
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    frames: List = []
    for snapshot in snapshots:
        ax.clear()
        _draw_snapshot_frame(
            ax, snapshot, colors, num_machines,
            x_max=x_max, title_prefix=title_prefix,
        )
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[..., :3].copy(), mode='RGB'))
    plt.close(fig)

    # Build a shared palette from the last frame, which contains every job
    # color that ever appears (earlier frames are subsets). Using frame 0 would
    # bake a too-small palette and cause later frames to collapse to grays.
    master = frames[-1].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [
        f.quantize(palette=master, dither=Image.Dither.NONE)
        for f in frames
    ]
    quantized[0].save(
        output_path,
        save_all=True,
        append_images=quantized[1:],
        duration=int(frame_duration_ms),
        loop=0,
        optimize=True,
    )
