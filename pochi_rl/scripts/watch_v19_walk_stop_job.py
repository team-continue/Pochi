"""Read training status over SSH and fetch verified artifacts when complete.

Uses the already authorized host helper; never issues motor or service commands.
Run with system Python, which owns the existing pexpect SSH helper environment.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REMOTE='srg@192.168.201.40'
REMOTE_ROOT='/home/srg/ws/pochi_walk_stop_20261013'
HELPER=Path('/tmp/pochi_training_20140.py')
LOCAL=Path(__file__).resolve().parents[1]/'logs/hardware_robustness/2026-10-13_v19_walk_stop'


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def remote_command(*args):
    result=subprocess.run([sys.executable,str(HELPER),*args],capture_output=True,text=True,timeout=210)
    if result.returncode:
        raise RuntimeError(f'SSH helper exited {result.returncode}: {result.stdout[-1500:]}')
    return result.stdout


def status():
    result=remote_command('ssh',REMOTE,'python3 /tmp/status_walk_stop.py')
    rows=[row.strip() for row in result.splitlines() if row.startswith('POCHI_STATUS_JSON=')]
    if len(rows)!=1:raise RuntimeError('Missing structured remote status')
    state=json.loads(rows[0].split('=',1)[1])
    write_json(LOCAL/'latest_status.json',state)
    return state


def retrieve(state):
    artifacts=LOCAL/'artifacts'
    artifacts.mkdir(exist_ok=True)
    saved=[]
    for remote_name,identity in state['artifacts'].items():
        # All names come from the task-owned remote status script. Reject paths
        # escaping its run before passing a path to scp.
        relative=Path(remote_name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unexpected artifact path')
        if remote_name=='training/run/run.json':local_name='run.json'
        elif remote_name=='training/run/progress.json':local_name='progress.json'
        else:local_name=relative.name
        destination=artifacts/local_name
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest()==identity['sha256']:
            saved.append(local_name)
            continue
        temporary=destination.with_suffix(destination.suffix+'.partial')
        remote_command('scp',REMOTE+':'+REMOTE_ROOT+'/'+remote_name,str(temporary))
        if hashlib.sha256(temporary.read_bytes()).hexdigest()!=identity['sha256']:
            raise RuntimeError('Artifact digest differs: '+remote_name)
        temporary.replace(destination)
        saved.append(local_name)
    contract=json.loads((artifacts/'policy_contract.json').read_text())
    assert contract['task']=='Pochi-V19-Walk-Stop-20261013'
    assert contract['completed_updates']==4000
    assert contract['model_sha256']==hashlib.sha256((artifacts/'policy.torchscript.pt').read_bytes()).hexdigest()
    guard=contract['safety_contract']['required_target_guard']
    assert guard['walk_estimated_torque_limit_nm']==16
    assert guard['walk_target_rate_limit_rad_s']==6
    write_json(LOCAL/'completion.json',dict(
        status='completed_and_downloaded',completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        remote_root=REMOTE_ROOT,files=saved,all_remote_sha256_verified=True,
        real_robot_deployment_performed=False))
    print('Completed and downloaded '+', '.join(saved),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    assert HELPER.is_file(), 'Existing authorized SSH helper is required'
    LOCAL.mkdir(parents=True,exist_ok=True)
    while True:
        try:
            state=status()
            supervisor=state.get('supervisor') or {}
            print(json.dumps({'status':supervisor.get('status'),
                'active_stage':supervisor.get('active_stage'),
                'completed_updates':(state.get('progress') or {}).get('completed_updates'),
                'checked_at':state['checked_at']}),flush=True)
            if supervisor.get('status')=='completed':
                retrieve(state)
                return 0
            if supervisor.get('status')=='failed':
                write_json(LOCAL/'completion.json',dict(status='remote_pipeline_failed',remote_status=supervisor))
                return 1
            if state.get('active_process_alive') is False:
                raise RuntimeError('Remote active process exited without a final stage record')
        except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
            print(f'Status/retrieval retry: {exc}',flush=True)
            write_json(LOCAL/'watcher_last_error.json',dict(
                error=str(exc),at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            if args.once:return 2
        if args.once:return 0
        time.sleep(60)


if __name__=='__main__':
    raise SystemExit(main())
