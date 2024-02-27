import os
import argparse


class Param:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):
        # common parameters
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--dataname", default="FewRel", type=str, help="Use TACRED or FewRel datasets.")
        parser.add_argument("--logname", default=None, type=str)
        parser.add_argument("--task_name", default="FewRel", type=str)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--total_rounds", default=5, type=int)

        # training parameters
        parser.add_argument("--bert_size", default="base", type=str)
        parser.add_argument("--batch_size", default=16, type=int)
        parser.add_argument("--num_tasks", default=10)
        parser.add_argument("--rel_per_task", default=8)
        parser.add_argument("--pattern", default="entity_marker")
        parser.add_argument("--max_length", default=256, type=int)
        parser.add_argument("--encoder_output_size", default=768, type=int)
        parser.add_argument("--vocab_size", default=30522, type=int)
        parser.add_argument("--marker_size", default=4, type=int)
        parser.add_argument("--num_workers", default=0, type=int)
        parser.add_argument("--encoder_checkpoint", action="store_true", default=False)
        parser.add_argument("--encoder_seprarate_embeddings", action="store_false", default=True)

        # MTL
        parser.add_argument("--mtl", default=None, type=str)
        parser.add_argument("--mtl_lr", default=1e-4, type=float)
        parser.add_argument("--tasktype", default=None, type=str)
        parser.add_argument("--c", default=0.5, type=float)
        parser.add_argument("--ETF", action="store_true", default=False)

        # learning rate
        parser.add_argument("--classifier_lr", default=1e-2, type=float)
        parser.add_argument("--encoder_lr", default=1e-3, type=float)
        parser.add_argument("--prompt_pool_lr", default=1e-3, type=float)
        # momentum
        parser.add_argument("--sgd_momentum", default=0.1, type=float)

        # generative
        parser.add_argument("--generative", default="GMM", type=str)
        parser.add_argument("--gmm_num_components", default=1, type=int)
        parser.add_argument("--gen_epochs", default=50, type=int)
        parser.add_argument("--gen_lr", default=1e-5, type=float)

        # loss balancing
        parser.add_argument("--pull_constraint_coeff", default=0.1, type=float)

        # epochs
        parser.add_argument("--classifier_epochs", default=100, type=int)
        parser.add_argument("--encoder_epochs", default=10, type=int)
        parser.add_argument("--prompt_pool_epochs", default=10, type=int)

        # replay size
        parser.add_argument("--replay_s_e_e", default=256, type=int)
        parser.add_argument("--replay_epochs", default=100, type=int)
        parser.add_argument("--replay_ratio", default=1.0, type=float)

        # seed
        parser.add_argument("--seed", default=2021, type=int)

        # max gradient norm
        parser.add_argument("--max_grad_norm", default=10.0, type=float)

        # dataset path
        parser.add_argument("--data_path", default="datasets/", type=str)
        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="../models/bert-base-uncased", type=str)

        # swag params
        parser.add_argument("--cov_mat", action="store_false", default=True)
        parser.add_argument("--max_num_models", type=int, default=10)
        parser.add_argument("--sample_freq", type=int, default=5)

        # prompt params
        parser.add_argument("--prompt_length", type=int, default=1)
        parser.add_argument("--prompt_embed_dim", type=int, default=768)
        parser.add_argument("--prompt_pool_size", type=int, default=80)
        parser.add_argument("--prompt_top_k", type=int, default=8)
        parser.add_argument("--prompt_init", default="uniform", type=str)
        parser.add_argument("--prompt_key_init", default="uniform", type=str)

        return parser


SIZE_DIM = {
    "base": 768,
    "large": 1024
}
