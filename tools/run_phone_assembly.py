"""Run the programmed contact simulation without any hardware adapter."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from phone_assembly.scene import scenario
from phone_assembly.worker import run_headless
from host_arm_ik import _resolve_urdf


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--randomize',action='store_true');parser.add_argument('--robot-ik',action='store_true')
    parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    result=run_headless(scenario(args.seed,args.randomize),args.output,_resolve_urdf(),args.robot_ik,args.render)
    print(json.dumps({k:result[k] for k in ('state','message','goal','hardware_commands','wall_seconds')},ensure_ascii=False))
    raise SystemExit(0 if result['state']=='complete' and result['goal']['success'] else 1)
