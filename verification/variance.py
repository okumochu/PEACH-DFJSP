"""Reconstruct static scheduling trajectories and variance statistics from inputs.

The checker uses neither the environment nor its lower-bound implementation.
"""
import math
import statistics
from verification.schedules import verify as verify_schedule,need,f32


def replay(fixture,actions,*,random_seed=None):
    st=fixture['instance']['static'];lengths=st['job_length'][0];pts=st['op_pt'][0]
    weights=list(map(f32,st['job_weight'][0]));dues=list(map(f32,st['job_due_date'][0]))
    nm=len(pts[0]);scale=max(max(row) for row in pts);need(scale>0,'Nonpositive scale')
    starts=[];offset=0
    for n in lengths:starts.append(offset);offset+=n
    need(len(actions)==offset,'Incomplete action trajectory')
    progress=[0]*len(lengths);job_end=[0.]*len(lengths);machine_end=[0.]*nm;schedule=[];trace=[]
    rng=None
    if random_seed is not None:
        import numpy as np
        rng=np.random.default_rng(random_seed)
    for depth in range(len(actions)+1):
        partial=(.5*max(machine_end)+.5*sum(w*max(0.,t-d) for j,(w,t,d) in enumerate(zip(weights,job_end,dues)) if progress[j]==lengths[j]))/scale
        estimates=[]
        for j,n in enumerate(lengths):
            ct=job_end[j]
            for k in range(progress[j],n):
                row=pts[starts[j]+k];ct=min(max(ct,machine_end[m])+p for m,p in enumerate(row) if p>0)
            estimates.append(ct)
        bound=(.5*max(estimates)+.5*sum(w*max(0.,ct-d) for w,ct,d in zip(weights,estimates,dues)))/scale
        trace.append(dict(depth=depth,partial=partial,dp=bound))
        if depth==len(actions):break
        a=actions[depth];need(type(a) is int and 0<=a<len(lengths)*nm,'Invalid action')
        j,m=divmod(a,nm);need(progress[j]<lengths[j],'Action after job completion')
        op_id=starts[j]+progress[j];pt=pts[op_id][m];need(pt>0,'Ineligible machine action')
        if rng is not None:
            feasible=[jj*nm+mm for jj,n in enumerate(lengths) if progress[jj]<n for mm,v in enumerate(pts[starts[jj]+progress[jj]]) if v>0]
            need(a==int(rng.choice(feasible)),'Random action differs from frozen uniform RNG')
        start=max(job_end[j],machine_end[m]);end=start+pt
        schedule.append(dict(op_id=op_id,job_id=j,machine_id=m,start_time=start,end_time=end,processing_time=pt))
        progress[j]+=1;job_end[j]=machine_end[m]=end
    return dict(scale=scale,trace=trace,schedule=schedule,terminal=trace[-1]['partial'])


def verify(record,fixture):
    checked=verify_schedule(record,fixture)
    independent=replay(fixture,record['actions'],random_seed=record['action_seed'] if record['policy']=='random' else None)
    need(record['process_time_scale']==independent['scale'],'Reward scale differs')
    need(len(record['trace'])==len(independent['trace']),'Trace depth coverage')
    indexed={op['op_id']:op for op in record['schedule']}
    for expected in independent['schedule']:
        actual=indexed[expected['op_id']]
        need(all(math.isclose(actual[k],v,rel_tol=1e-7,abs_tol=1e-4) for k,v in expected.items()),'Action/schedule mismatch')
    terminal=record['objective']/independent['scale'];errors=[]
    for saved,expected in zip(record['trace'],independent['trace']):
        need(saved['depth']==expected['depth'],'Depth index')
        for field in ['partial','dp']:
            error=abs(saved[field]-expected[field]);errors.append(error)
            need(math.isfinite(saved[field]) and math.isclose(saved[field],expected[field],rel_tol=1e-5,abs_tol=1e-4),'Independent '+field+' reconstruction differs')
        tol=1e-5*max(1.,abs(terminal))
        need(saved['dp']+tol>=saved['partial'] and saved['dp']<=terminal+tol,'Bound admissibility/tightness')
    need(math.isclose(record['terminal_normalized_objective'],terminal,rel_tol=1e-5,abs_tol=1e-4),'Terminal target')
    need(len(record['dp_rewards'])==len(record['actions']),'Reward/action length')
    for i,reward in enumerate(record['dp_rewards']):
        need(math.isclose(reward,record['trace'][i]['dp']-record['trace'][i+1]['dp'],rel_tol=1e-5,abs_tol=1e-4),'Reward difference')
    checked.update(depths=len(record['actions']),max_trace_reconstruction_error=max(errors))
    return checked


def depth_statistics(partial,bound,terminal):
    need(len(partial)==len(bound)==len(terminal) and len(partial)>=2,'Fixed-depth population')
    gap=[l-o for o,l in zip(partial,bound)];residual=[t-l for t,l in zip(terminal,bound)];objres=[t-o for t,o in zip(terminal,partial)]
    vg=statistics.pvariance(gap);ve=statistics.pvariance(residual);vo=statistics.pvariance(objres)
    mg,me=statistics.mean(gap),statistics.mean(residual);cov=statistics.mean((g-mg)*(e-me) for g,e in zip(gap,residual))
    lhs=vo-ve;rhs=vg+2*cov;error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
    need(error<=1e-8,'Variance identity mismatch')
    return dict(n=len(partial),var_gap=vg,covariance=cov,var_return_partial=vo,var_return_dp=ve,
        condition_holds=cov>-.5*vg,variance_reduced=vo>ve,variance_ratio=vo/ve if ve>0 else None,
        identity_relative_error=error)
