"""Offline transport of actual frozen 39D/4D storage. No simulator or optimizer.

Two persistent spawn workers replay disjoint shards of one real rollout; this
measures communication, NOT live synchronized PhysX or independent sampling.
"""
import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import time
import numpy as np
import psutil
import torch

def arrays(x):
    if isinstance(x,torch.Tensor): return x.numpy().copy()
    if isinstance(x,dict): return {k:arrays(v) for k,v in x.items()}
    return x

def equal(a,b):
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    return a==b

def shard(data,i):
    result={}
    for k,v in data.items():
        if k=='version': result[k]=v
        elif isinstance(v,dict): result[k]={key:(x[i*128:(i+1)*128].copy() if k=='last_observations' else x[:,i*128:(i+1)*128].copy()) for key,x in v.items()}
        else: result[k]=v[:,i*128:(i+1)*128].copy()
    return result

def validate_envelope(packet, version, round_id, worker_id):
    if (packet['version'] != version or packet['rollout']['version'] != version
            or packet['round'] != round_id or packet['worker'] != worker_id):
        raise ValueError('stale, mixed, or wrongly addressed rollout')


def worker(conn,path,index):
    torch.set_num_threads(1)
    data=arrays(torch.load(path,weights_only=False));policy=data.pop('policy');part=shard(data,index)
    conn.send(('ready',psutil.Process().memory_info().rss))
    while True:
        packet=conn.recv_bytes()
        if packet==b'stop': break
        start=time.perf_counter();request=pickle.loads(packet);decode=time.perf_counter()-start
        assert request['version']==part['version'] and equal(request['policy'],policy)
        ack=time.perf_counter();conn.send(('ack',ack,decode))
        start=time.perf_counter();payload=pickle.dumps(dict(version=part['version'],round=request['round'],worker=index,rollout=part),protocol=5)
        encode=time.perf_counter()-start
        conn.send_bytes(payload)
        conn.send(dict(encode_s=encode,ready_at=ack,payload_bytes=len(payload),rss=psutil.Process().memory_info().rss))
    conn.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1)
    data=arrays(torch.load(a.source,weights_only=False));policy=data.pop('policy')
    assert data['observations']['policy'].shape==(48,256,39) and data['actions'].shape==(48,256,4)
    ctx=mp.get_context('spawn');connections=[];workers=[];start=time.perf_counter()
    for i in range(2):
        left,right=ctx.Pipe();proc=ctx.Process(target=worker,args=(right,a.source,i));proc.start();right.close();connections.append(left);workers.append(proc)
    ready=[c.recv() for c in connections];startup=time.perf_counter()-start
    rows=[]
    try:
        for r in range(6):
            start=time.perf_counter();packet=pickle.dumps(dict(version=data['version'],round=r,policy=policy),protocol=5);encode=time.perf_counter()-start
            start=time.perf_counter()
            for c in connections:c.send_bytes(packet)
            ack=[c.recv() for c in connections];broadcast=time.perf_counter()-start
            start=time.perf_counter();parts=[];decode=0.;received=[]
            for i,c in enumerate(connections):
                blob=c.recv_bytes();t=time.perf_counter();part=pickle.loads(blob);decode+=time.perf_counter()-t
                validate_envelope(part,data['version'],r,i)
                parts.append(part['rollout']);received.append(time.perf_counter())
            transfer=time.perf_counter()-start;stats=[c.recv() for c in connections]
            t=time.perf_counter()
            merged={}
            for k,v in data.items():
                if isinstance(v,dict): merged[k]={key:np.concatenate([q[k][key] for q in parts],axis=0 if k=='last_observations' else 1) for key in v}
                elif isinstance(v,np.ndarray):merged[k]=np.concatenate([q[k] for q in parts],axis=1)
                else:merged[k]=v
            merge=time.perf_counter()-t
            assert equal(merged,data)
            rows.append(dict(round=r,policy_bytes=len(packet),policy_encode_s=encode,broadcast_ack_s=broadcast,
                rollout_receive_decode_s=transfer,learner_decode_s=decode,merge_s=merge,workers=stats,
                readiness_skew_s=max(x[1] for x in ack)-min(x[1] for x in ack),receive_skew_s=max(received)-min(received),
                learner_rss=psutil.Process().memory_info().rss))
        # Reject wrong version/round before admitting data to a global storage.
        rejected=0
        for bad in [dict(part,version='wrong'),dict(part,round=4),dict(part,worker=99)]:
            try:validate_envelope(bad,data['version'],5,1)
            except ValueError:rejected+=1
        result=dict(passed=rejected==3,scope='offline actual storage transport, no live synchronized sampler',
            startup_s=startup,ready=ready,source=str(a.source),shapes={k:list(v.shape) for k,v in data.items() if isinstance(v,np.ndarray)},rounds=rows,
            steady_median_total_s=float(np.median([x['policy_encode_s']+x['broadcast_ack_s']+x['rollout_receive_decode_s']+x['merge_s'] for x in rows[1:]])))
        a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('rounds','shapes')}))
    finally:
        for c in connections:c.send_bytes(b'stop')
        for proc in workers:
            proc.join(10)
            if proc.is_alive():proc.kill();proc.join()
        for c in connections:c.close()

if __name__=='__main__':main()
