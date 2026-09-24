"""Independent runs; no architecture/hyperparameter tuning on special inputs."""
import argparse
from train import run

QUEUES={
 'baseline':[('gru_clean',17),('gru_aug',17),('mlp',17),('text',17),('gru_aug',29),('gru_aug',43),('mlp',29),('mlp',43)],
 'state':[('state_aug',17),('state_clean',17),('state_no_gap',17),('state_no_history',17),('smooth_aug',17),('state_aug',29),('state_aug',43),('smooth_aug',29),('smooth_aug',43)],
 'clean_repeats':[('gru_clean',29),('gru_clean',43)],
}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--queue',choices=QUEUES,required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for config,seed in QUEUES[args.queue]:
        run(config,seed,args.device)
