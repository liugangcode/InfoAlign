import yaml
import argparse

model_hyperparams = [
    "emb_dim",
    "model",
    "num_layer",
    "readout",
    "norm_layer",
    "threshold",
    "walk_length",
    "prior",
]

def load_arguments_from_yaml(filename, model_only=False):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
    if model_only:
        config = {k: v for k, v in config.items() if k in model_hyperparams}
    else:
        config = yaml.safe_load(file)
    return config


def save_arguments_to_yaml(args, filename, model_only=False):
    if model_only:
        args = {k: v for k, v in vars(args).items() if k in model_hyperparams}
    else:
        args = vars(args)

    with open(filename, "w") as f:
        yaml.dump(args, f)


def get_args():
    parser = argparse.ArgumentParser(
        description="Learning molecular representation in a cell"
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="number of workers for data loader"
    )
    parser.add_argument(
        "--no-print", action="store_true", default=False, help="don't use progress bar"
    )

    parser.add_argument("--dataset", default="pretrain", type=str, help="dataset name")

    # model
    parser.add_argument(
        "--model",
        type=str,
        default="gin-virtual",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)",
    )
    parser.add_argument(
        "--readout", type=str, default="sum", help="graph readout (default: sum)"
    )
    parser.add_argument(
        "--norm-layer",
        type=str,
        default="batch_norm",
        help="GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)",
    )
    parser.add_argument(
        "--drop-ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--num-layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=300,
        help="dimensionality of hidden units in GNNs (default: 300)",
    )
    # training
    ## pretraining
    parser.add_argument(
        "--walk-length",
        type=int,
        default=4,
        help="pretraining context length",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="minimum similarity threshold for context graph",
    )
    # prior
    parser.add_argument(
        "--prior",
        type=float,
        default=1e-9,
        help="loss weight to prior",
    )

    ## other
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5120,
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument("--wdecay", default=1e-5, type=float, help="weight decay")
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of epochs to train"
    )
    parser.add_argument(
        "--initw-name",
        type=str,
        default="default",
        help="method to initialize the model paramter",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ckpt/pretrain.pt",
        help="path to the pretrained model",
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="patience for early stop"
    )

    args = parser.parse_args()
    print("no print", args.no_print)

    ## n_steps for solver
    args.n_steps = 1
    return args
