"""Independent segment checks for the retained dynamic scenario fixtures.

No training-environment event or boundary helpers are imported here.
"""
import copy
import math
from verification.schedules import verify as verify_static, need, f32


def unpack(value):
    if isinstance(value,dict):
        if set(value)=={'dtype','shape','data'}:return value['data']
        return {k:unpack(v) for k,v in value.items()}
    if isinstance(value,list):return [unpack(v) for v in value]
    return value


def groups(raw):
    e=raw['dynamic_events']
    queue=[dict(type='job_arrival',time=float(t),job_idx=i) for i,t in enumerate(e['new_job_arrival'])]
    queue += [dict(type='machine_breakdown',time=float(b['time']),machine_idx=int(b['machine_idx']),duration=float(b['duration'])) for b in e['machine_breakdown']]
    for change in e['priority_change']:
        need(isinstance(change,dict) and change.get('mode')=='set' and isinstance(change.get('job_indices'),list),
             'Dynamic scenario contract requires explicit set-priority events')
        queue.append(dict(type='priority_change',time=float(change['time']),job_indices=list(change['job_indices']),
                          new_weight=float(change['new_weight']),mode='set'))
    queue.sort(key=lambda r:r['time']);out=[(0.,None)]
    for event in queue:
        if out[-1][1] is not None and out[-1][0]==event['time']:out[-1][1].append(event)
        else:out.append((event['time'],[event]))
    return out


def same_numbers(a,b):
    if isinstance(b,dict):return isinstance(a,dict) and a.keys()==b.keys() and all(same_numbers(a[k],v) for k,v in b.items())
    if isinstance(b,list):return isinstance(a,list) and len(a)==len(b) and all(same_numbers(x,y) for x,y in zip(a,b))
    if isinstance(b,(float,int)):return isinstance(a,(float,int)) and math.isfinite(a) and math.isclose(a,b,abs_tol=1e-3,rel_tol=1e-6)
    return a==b


def verify(record,fixture):
    raw=record['raw'];data=fixture['instance'];st=copy.deepcopy(data['static']);dyn=data['dynamic_jobs']
    snaps=raw['snapshots'];events=groups(data);need(len(snaps)==len(events)==raw['num_segments'],'Missing event-segment plans')
    need(raw['schedule']==raw['schedule_data']['schedule']==snaps[-1]['schedule'],'Final schedule differs')
    releases=[0.]*len(st['job_length'][0]);outages=[];fragments=[];previous=None;placements=0
    for index,(snap,(now,trigger)) in enumerate(zip(snaps,events)):
        need(snap['segment_idx']==index and snap['triggering_event_set']==trigger and snap['time_now']==now,'Event identity/order')
        fixed={}
        if trigger:
            broken={e['machine_idx'] for e in trigger if e['type']=='machine_breakdown'}
            for op in previous:
                if op['start_time']>=now:continue
                if op['machine_id'] in broken and op['end_time']>now:
                    fragments.append(dict(env_idx=0,op_idx=op['op_id'],machine_idx=op['machine_id'],start_time=op['start_time'],interrupt_time=now))
                else:fixed[op['op_id']]=op
            for event in trigger:
                if event['type']=='machine_breakdown':
                    outages.append(dict(env_idx=0,machine_idx=event['machine_idx'],start_time=now,end_time=now+event['duration']))
                elif event['type']=='job_arrival':
                    j=event['job_idx'];n=dyn['job_length'][j];start=sum(dyn['job_length'][:j]);st['job_length'][0].append(n)
                    st['op_pt'][0].extend(dyn['op_pt'][start:start+n])
                    for key in ['job_weight','job_due_date']:st[key][0].append(f32(dyn[key][j]))
                    releases.append(f32(now))
                else:
                    need(event['type']=='priority_change','Unknown event')
                    for j in event['job_indices']:
                        need(type(j) is int and 0<=j<len(st['job_weight'][0]),'Priority target not yet present')
                        st['job_weight'][0][j]=f32(event['new_weight'])
        need(sorted(snap['fixed_op_ids'])==sorted(fixed),'Fixed operation set')
        need(same_numbers(snap['breakdown_periods'],outages),'Outage history')
        need(same_numbers(snap['interrupted_ops'],fragments),'Interruption history')
        jobs=[];offset=0
        for j,n in enumerate(st['job_length'][0]):
            jobs.append(dict(job_idx=j,weight=f32(st['job_weight'][0][j]),due_date=f32(st['job_due_date'][0][j]),release_time=releases[j],first_op=offset,last_op=offset+n-1));offset+=n
        need(same_numbers(snap['job_data'],jobs),'Job data differs from exact input/arrival history')
        schedule=snap['schedule'];indexed={op['op_id']:op for op in schedule}
        ends=[max(op['end_time'] for op in schedule if op['job_id']==j) for j in range(len(jobs))]
        twt=sum(j['weight']*max(0.,end-j['due_date']) for j,end in zip(jobs,ends));mk=max(ends)
        probe=dict(schedule=schedule,makespan=snap['segment_makespan'],twt=twt,objective=.5*(mk+twt),measured_seconds=record['measured_seconds'])
        check=verify_static(probe,dict(family='static',instance=dict(static=st,dynamic_events={},dynamic_jobs={})))
        placements+=check['operations'];occupied={}
        for op in schedule:
            machine=op['machine_id'];a,b=op['start_time'],op['end_time']
            need(a+1e-3>=releases[op['job_id']],'Operation before release')
            if op['op_id'] in fixed:need(same_numbers(op,fixed[op['op_id']]),'Changed fixed operation')
            else:need(a+1e-3>=now,'Unfixed operation scheduled before event')
            for bd in outages:
                if machine==bd['machine_idx']:need(b<=bd['start_time']+1e-3 or a+1e-3>=bd['end_time'],'Operation overlaps breakdown')
            occupied.setdefault(machine,[]).append((a,b))
        for frag in fragments:occupied.setdefault(frag['machine_idx'],[]).append((frag['start_time'],frag['interrupt_time']))
        for spans in occupied.values():
            spans.sort();need(all(a[1]<=b[0]+1e-3 for a,b in zip(spans,spans[1:])),'Machine overlap including discarded work')
        previous=schedule
    need(same_numbers(raw['breakdown_periods'],outages) and same_numbers(raw['interrupted_ops'],fragments),'Final event history')
    need(same_numbers(raw['schedule_data']['job_data'],jobs),'Final job metadata')
    need(same_numbers(raw['schedule_data']['breakdown_periods'],outages) and same_numbers(raw['schedule_data']['interrupted_ops'],fragments),'Final schedule-data event history')
    for k,expected in [('makespan',mk),('twt',twt),('objective',.5*(mk+twt))]:
        need(math.isclose(record[k],expected,rel_tol=2e-6,abs_tol=.002) and record[k]==raw[k],'Final '+k+' differs')
    need(raw['schedule_data']['num_jobs']==len(jobs) and raw['schedule_data']['num_machines']==len(st['op_pt'][0][0]),'Final dimensions')
    return dict(valid=True,operations=len(previous),segment_operation_placements=placements,segments=len(snaps),interrupted_fragments=len(fragments),makespan=mk,twt=twt,objective=.5*(mk+twt))
