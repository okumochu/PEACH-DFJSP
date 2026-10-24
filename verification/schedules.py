"""Independent raw-input verifier for September 25 static/Public evaluations."""
import hashlib
import math
import struct


def need(value, message):
    if not value:
        raise ValueError(message)


def parse_fjs(text):
    lines = [line.split() for line in text.splitlines() if line.strip()]
    jobs, machines = map(int, lines[0][:2])
    need(len(lines) == jobs + 1, 'FJS job count')
    lengths, processing = [], []
    for line in lines[1:]:
        numbers = list(map(int, line)); lengths.append(numbers[0]); i = 1
        for _ in range(numbers[0]):
            count = numbers[i]; i += 1; row = [0] * machines
            need(count > 0, 'No eligible machine')
            for _ in range(count):
                machine, duration = numbers[i:i+2]; i += 2
                need(1 <= machine <= machines and duration > 0 and row[machine-1] == 0,
                     'Invalid FJS alternative')
                row[machine-1] = duration
            processing.append(row)
        need(i == len(numbers), 'Unparsed FJS fields')
    return lengths, processing


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def verify(record, fixture):
    if fixture['family'] == 'public':
        problem = fixture['problem']; text = problem['fjs_text']
        need(hashlib.sha256(text.encode()).hexdigest() == problem['sha256'], 'FJS hash')
        lengths, pts = parse_fjs(text)
        need(lengths == problem['job_length'] and pts == problem['processing_times'], 'FJS arrays')
        weights, dues, mw, tw = [0.] * len(lengths), [0.] * len(lengths), 1., 0.
    else:
        need(fixture['family'] == 'static', 'Unsupported evaluation family')
        data = fixture['instance']
        need(not any(data['dynamic_events'].values()), 'Static fixture has events')
        need(not any(data.get('dynamic_jobs', {}).values()), 'Static fixture has arriving jobs')
        st = data['static']; lengths = st['job_length'][0]; pts = st['op_pt'][0]
        weights = list(map(f32, st['job_weight'][0])); dues = list(map(f32, st['job_due_date'][0]))
        mw, tw = .5, .5
    need(lengths and all(type(n) is int and n > 0 for n in lengths), 'Job lengths')
    need(len(pts) == sum(lengths), 'Operation count')
    need(len(weights) == len(dues) == len(lengths), 'Job objective data')
    nm = len(pts[0]); jobs = [j for j, n in enumerate(lengths) for _ in range(n)]
    need(all(len(p) == nm for p in pts), 'Machine count')
    ops, machines = {}, {}
    for op in record['schedule']:
        i, m = op['op_id'], op['machine_id']; start, end = op['start_time'], op['end_time']
        need(type(i) is int and 0 <= i < len(jobs) and i not in ops, 'Duplicate/invalid operation')
        need(type(m) is int and 0 <= m < nm and op['job_id'] == jobs[i], 'Operation assignment')
        duration = f32(pts[i][m])
        need(duration > 0 and all(math.isfinite(x) for x in (start, end, duration, op['processing_time'])), 'Duration')
        need(start >= -1e-4 and abs(end-start-duration) <= 1e-3 and
             abs(op['processing_time']-duration) <= 1e-3, 'Processing time mismatch')
        ops[i] = op; machines.setdefault(m, []).append((start, end))
    need(set(ops) == set(range(len(jobs))), 'Incomplete schedule')
    completion = [0.] * len(lengths)
    for i, j in enumerate(jobs):
        op = ops[i]
        need(not i or jobs[i-1] != j or ops[i-1]['end_time'] <= op['start_time'] + 1e-4,
             'Precedence violation')
        completion[j] = max(completion[j], op['end_time'])
    for intervals in machines.values():
        intervals.sort()
        need(all(a[1] <= b[0]+1e-4 for a,b in zip(intervals, intervals[1:])), 'Machine overlap')
    makespan = max(completion)
    twt = sum(w * max(0., end-due) for w,end,due in zip(weights,completion,dues))
    objective = mw * makespan + tw * twt
    for key, expected in [('makespan', makespan), ('twt', twt), ('objective', objective)]:
        value = record[key]
        need(math.isfinite(value) and math.isclose(value, expected, rel_tol=2e-6, abs_tol=.002),
             f'Recomputed {key} differs: {value} vs {expected}')
    need(math.isfinite(record['measured_seconds']) and record['measured_seconds'] > 0, 'Measured timing')
    return dict(valid=True, operations=len(ops), makespan=makespan, twt=twt, objective=objective)
