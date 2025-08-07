import glob
import os


def main():
    os.chdir(os.path.expanduser('~/alpha-seed/'))
    os.system('cp ~/seed_rl/tasks/rl/verifier/* ~/alpha-seed/alpha_seed/utils/reward_score/vlm_verifiers/')
    os.system('bash scripts/format.sh')
    for file in glob.glob(os.path.expanduser('~/alpha-seed/alpha_seed/utils/reward_score/vlm_verifiers/*.py')):
        print('Processing file: ', file, '...')
        if os.path.abspath(file) != os.path.abspath(__file__):
            with open(file, 'r') as fin:
                text = fin.read()
                text = text.replace('tasks.rl.verifier.', 'alpha_seed.utils.reward_score.vlm_verifiers.')
                with open(file, 'w') as fout:
                    fout.write(text)


if __name__ == '__main__':
    main()
