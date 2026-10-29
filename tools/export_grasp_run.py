"""Export one recorded execution to synchronized tabular data, without robot I/O.

This is an audit/training interchange CSV, not a claimed LeRobot dataset.
The immutable input NPZ is referenced as a snapshot, never duplicated as live video.
"""
import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def export(directory,output):
    directory,output=Path(directory),Path(output)
    request=json.loads((directory/'request.json').read_text())
    rows=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines() if line.strip()]
    if not rows:raise ValueError('运行记录为空')
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('导出目录必须为空，避免覆盖其他运行记录')
    for name in ('request.json','environment.json','observation_snapshot.npz'):
        source=directory/name
        if source.is_file():
            try:os.link(source,output/name)
            except OSError:shutil.copyfile(source,output/name)
    fields=['timestamp','state','trajectory_elapsed_s']+[f'observation.q{i}' for i in range(16)]+[f'action.q{i}' for i in range(16)]
    for side in ('left','right'):
        fields += [f'{side}.{axis}' for axis in ('Fx','Fy','Fz','Mx','My','Mz')]
        fields += [f'{side}.hand{i}' for i in range(6)]
        fields += [f'{side}.command_hand{i}' for i in range(6)]
    with (output/'episode.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for row in rows:
            data={'timestamp':row['stamp'],'state':row['state'],'trajectory_elapsed_s':row.get('elapsed',0.)}
            for prefix,values in (('observation',(row.get('feedback') or {}).get('q16',[])),('action',row.get('command_q16',[]))):
                data.update({f'{prefix}.q{i}':value for i,value in enumerate(values)})
            for side in ('left','right'):
                wrench=(row.get('wrenches') or {}).get(side) or {}
                data.update({f'{side}.{axis}':value for axis,value in zip(('Fx','Fy','Fz','Mx','My','Mz'),
                    (wrench.get('force') or [])+(wrench.get('torque') or []))})
                hand=row.get('hands',{}).get(side,[])
                if isinstance(hand,dict):hand=hand.get('positions',[])
                data.update({f'{side}.hand{i}':value for i,value in enumerate(hand)})
                command=row.get('command_hand') or {}
                if command.get('side')==side:data.update({f'{side}.command_hand{i}':value for i,value in enumerate(command['positions'])})
            writer.writerow(data)
    metadata={'version':1,'source':str(directory.resolve()),'execution_mode':request['execution_mode'],
        'observation_id':request.get('observation_id'),'input_snapshot':'observation_snapshot.npz' if (output/'observation_snapshot.npz').exists() else None,
        'snapshot_is_video':False,'joint_units':['m']+['rad']*15,'force_units':'N','torque_units':'N m',
        'hand_units':'normalized_0_1','samples':len(rows),'final_state':rows[-1]['state'],
        'physical_task_success':None}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(export(args.directory,args.output),ensure_ascii=False))
