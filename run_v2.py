"""Resume the complete latest experiment. Existing completed folds are reused."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--skip-preparation',action='store_true'); parser.add_argument('--verify-media',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parent
    commands=[] if args.skip_preparation else ['prepare.py','prepare_v2.py']
    commands+=['train_v2.py','train_ensemble.py','report_v2.py','evaluate_missing_context.py','report_ensemble.py','test_v2.py']
    if args.verify_media: commands.append('verify_v2.py')
    for script in commands:
        print('Running',script,flush=True); subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)
if __name__=='__main__': main()
