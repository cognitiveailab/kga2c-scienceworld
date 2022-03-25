import argparse

from gdqn import KGA2CTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./logs/')
    parser.add_argument('--spm_file', default='./spm_models/unigram_8k.model')
    parser.add_argument('--tsv_file', default='../data/sciworld_entity2id.tsv')
    parser.add_argument('--rom_file_path')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--embedding_size', default=50, type=int)
    parser.add_argument('--hidden_size', default=100, type=int)
    parser.add_argument('--padding_idx', default=0, type=int)
    parser.add_argument('--gat_emb_size', default=50, type=int)
    parser.add_argument('--dropout_ratio', default=0.2, type=float)
    parser.add_argument('--preload_weights', default='')
    parser.add_argument('--bindings', default='zork1')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--steps', default=100000, type=int)
    parser.add_argument('--reset_steps', default=100, type=int)
    parser.add_argument('--stuck_steps', default=10, type=int)
    parser.add_argument('--trial', default='base')
    parser.add_argument('--loss', default='value_policy_entropy')
    parser.add_argument('--graph_dropout', default=0.0, type=float)
    parser.add_argument('--mask_dropout', default=0.5, type=float)
    parser.add_argument('--k_object', default=1, type=int)
    parser.add_argument('--g_val', default=False, type=bool)
    parser.add_argument('--entropy_coeff', default=0.03, type=float)
    parser.add_argument('--clip', default=40, type=int)
    parser.add_argument('--bptt', default=8, type=int)
    parser.add_argument('--value_coeff', default=9, type=float)
    parser.add_argument('--template_coeff', default=3, type=float)
    parser.add_argument('--object_coeff', default=9, type=float)
    parser.add_argument('--recurrent', default=True, type=bool)
    parser.add_argument('--checkpoint_interval', default=500, type=int)
    parser.add_argument('--test_interval', default=1000, type=int)
    parser.add_argument('--no-gat', dest='gat', action='store_false')
    parser.add_argument('--masking', default='interactive', choices=['kg', 'interactive', 'none'], help='Type of object masking applied')
    parser.add_argument('--task_num', default=0, type=int)
    parser.add_argument('--simplification_str', default='')
    parser.add_argument('--max_histories_per_file', type=int, default=1000)
    parser.add_argument('--max_eval_episodes', type=int, default=10)

    parser.set_defaults(gat=True)
    args = parser.parse_args()
    params = vars(args)
    return params


if __name__ == "__main__":
    params = parse_args()
    print(params)
    trainer = KGA2CTrainer(params)
    trainer.train(params['steps'])
