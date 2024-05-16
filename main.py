import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import math
import logging
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from configures.arguments import (
    load_arguments_from_yaml,
    save_arguments_to_yaml,
    get_args,
)
from dataset.create_datasets import get_data
from utils import validate, init_weights, save_prediction
from utils.train_funcs import pretrain_func, finetune_func


def get_logger(name, logfile=None):
    """create a nice logger"""
    logger = logging.getLogger(name)
    # clear handlers if they were created in other runs
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    # create console handler add add to logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler add add to logger when name is not None
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main(args, seed):
    device = torch.device("cuda", args.gpu_id)
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.dataset == "pretrain":
        dataset, context_graph = get_data(args, "./raw_data", transform="pyg")
        context_graph = context_graph[0]
    else:
        dataset = get_data(args, "./raw_data", transform="pyg")
        context_graph = None

    split_idx = dataset.get_idx_split()
    args.num_trained = len(split_idx["train"])
    args.task_type = dataset.task_type
    args.steps = args.num_trained // args.batch_size + 1

    train_loader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    if args.dataset == "pretrain":
        from models.gnn import GNN
        from torch.distributions import Normal, Independent

        test_loader = None
        model = GNN(
            gnn_type=args.model,
            num_tasks=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            graph_pooling=args.readout,
            norm_layer=args.norm_layer,
        ).to(device)
        init_weights(model, args.initw_name, init_gain=0.02)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

        prior_mu = torch.zeros(args.emb_dim).to(device)
        prior_sigma = torch.ones(args.emb_dim).to(device)
        args.prior_dist = Independent(Normal(loc=prior_mu, scale=prior_sigma), 1)

    elif args.dataset.startswith("finetune"):
        from models.gnn import FineTuneGNN

        valid_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model = FineTuneGNN(
            gnn_type=args.model,
            num_tasks=dataset.num_tasks,
            num_layer=args.num_layer,
            emb_dim=args.emb_dim,
            drop_ratio=args.drop_ratio,
            graph_pooling=args.readout,
            norm_layer=args.norm_layer,
        ).to(device)
        model.load_pretrained_graph_encoder(args.model_path)
        model.freeze_graph_encoder()
        optimizer = optim.Adam(
            model.task_decoder.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    else:
        raise ValueError("Invalid dataset name")

    # scheduler = None
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs * args.steps)
    
    logging.warning(f"device: {args.device}, " f"n_gpu: {args.n_gpu}, ")
    logger.info(dict(args._get_kwargs()))
    logger.info(model)
    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}@{args.num_trained}/{len(split_idx['valid'])}/{len(split_idx['test'])}"
    )
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Total train batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {args.epochs * args.steps}")

    train_loaders = {"train_iter": iter(train_loader), "train_loader": train_loader}

    best_train, best_valid, best_test, best_count = None, None, None, None
    best_epoch = 0
    loss_tots = []
    if args.dataset == "pretrain":  # later args.finetune
        for epoch in range(0, args.epochs):
            loss, train_loaders = pretrain_func(
                args, model, train_loaders, context_graph, optimizer, scheduler, epoch
            )
            loss_tots.append(loss)
            if epoch == args.epochs - 1:
                torch.save(model.state_dict(), args.model_path)
                yaml_path = args.model_path.replace(".pt", ".yaml")
                save_arguments_to_yaml(args, yaml_path, model_only=True)
                logger.info(
                    f"Finished Training \n Model saved at {args.model_path} and Arguments saved at {yaml_path} with loss {loss_tots}"
                )

    elif args.dataset.startswith("finetune"):
        args.task_type = (
            "regression" if "mae" in dataset.eval_metric else "classification"
        )
        best_params = None
        for epoch in range(0, args.epochs):
            train_loaders = finetune_func(
                args, model, train_loaders, optimizer, scheduler, epoch
            )
            valid_perf = validate(args, model, valid_loader)

            if epoch > 0:
                is_improved = (
                    valid_perf[dataset.eval_metric] < best_valid
                    if args.task_type == "regression"
                    else valid_perf[dataset.eval_metric] > best_valid
                )
            if epoch == 0 or is_improved:
                train_perf = validate(args, model, train_loader)
                test_perf = validate(args, model, test_loader)
                best_params = parameters_to_vector(model.parameters())
                best_valid = valid_perf[dataset.eval_metric]
                best_test = test_perf[dataset.eval_metric]
                best_train = train_perf[dataset.eval_metric]
                best_epoch = epoch
                best_count = test_perf.get("count", None)
                if best_count is None:
                    best_count = test_perf.get("mae_list", None)
                if not args.no_print:
                    logger.info(
                        "Update Epoch {}: best_train: {:.4f} best_valid: {:.4f}, best_test: {:.4f}".format(
                            epoch, best_train, best_valid, best_test
                        )
                    )
                    if best_count is not None and args.task_type == "classification":
                        outstr = "Best Count: "
                        for key, value in best_count.items():
                            sum_num = int(np.nansum(value))
                            nan_num = sum(np.isnan(value))
                            outstr += f"{key}: {sum_num/len(value):.4f} (nan {sum(np.isnan(value))} / {len(value)}), "
                        logger.info(outstr)
            else:
                if not args.no_print:
                    logger.info(
                        "Epoch {}: best_valid: {:.4f}, current_valid: {:.4f}, patience: {}/{}".format(
                            epoch,
                            best_valid,
                            valid_perf[dataset.eval_metric],
                            epoch - best_epoch,
                            args.patience,
                        )
                    )
                if epoch - best_epoch > args.patience:
                    break

        logger.info(
            "Finished. \n {}-{} Best validation epoch {} with metric {}, train {:.4f}, valid {:.4f}, test {:.4f}".format(
                args.dataset, args.pretrain_name, best_epoch, dataset.eval_metric, best_train, best_valid, best_test
            )
        )
        vector_to_parameters(best_params, model.parameters())
        save_prediction(model, device, test_loader, dataset, args.output_dir, seed)

    return (
        args.pretrain_name,
        args.dataset,
        dataset.eval_metric,
        best_train,
        best_valid,
        best_test,
        best_epoch,
        best_count,
    )


if __name__ == "__main__":
    import os
    import pandas as pd

    args = get_args()
    log_path = args.model_path.replace(".pt", ".log")

    pretrain_name = args.model_path.split("/")[-1]
    pretrain_name = pretrain_name.split(".")[0]
    args.pretrain_name = pretrain_name
    if args.dataset.startswith("finetune"):
        args.output_dir = f"output/{args.dataset}/{pretrain_name}"
        yaml_path = args.model_path.replace(".pt", ".yaml")
        config = load_arguments_from_yaml(yaml_path, model_only=True)
        for arg, value in config.items():
            setattr(args, arg, value)
        log_path = log_path + ".finetune"
    else:
        log_path = log_path + ".pretrain"

    # logger = get_logger(__name__, logfile=log_path)
    logger = get_logger(__name__)
    args.logger = logger
    print(vars(args))

    if args.dataset.startswith("pretrain"):
        main(args, 0)
    else:
        df = pd.DataFrame()
        for i in range(5):
            model, dataset, metric, train, valid, test, epoch, count = main(args, i)
            if "auc" in metric:
                new_results = {
                    "model": model,
                    "dataset": dataset,
                    "seed": i,
                    "metric": metric,
                    "train": train,
                    "valid": valid,
                    "test": test,
                    "epoch": epoch,
                    "suc_80": round(np.nansum(count[80]) / len(count[80]), 4),
                    "suc_85": round(np.nansum(count[85]) / len(count[85]), 4),
                    "suc_90": round(np.nansum(count[90]) / len(count[90]), 4),
                    "suc_95": round(np.nansum(count[95]) / len(count[95]), 4),
                    "thr_80": count[80],
                    "thr_85": count[85],
                    "thr_90": count[90],
                    "thr_95": count[95],
                }
            else:
                mae_list = count
                new_results = {
                    "model": model,
                    "dataset": dataset,
                    "seed": i,
                    "metric": metric,
                    "train": train,
                    "valid": valid,
                    "test": test,
                    "epoch": epoch,
                    "mae_1": mae_list[0],
                    "mae_2": mae_list[1],
                    "mae_3": mae_list[2],
                    "mae_4": mae_list[3],
                    "mae_5": mae_list[4],
                    "mae_6": mae_list[5],
                }
            df = pd.concat([df, pd.DataFrame([new_results])], ignore_index=True)

        summary_each = f"output/{args.dataset}/summary_each.csv"
        if os.path.exists(summary_each):
            df.to_csv(summary_each, mode="a", header=False, index=False)
        else:
            df.to_csv(summary_each, index=False)
        print(df)

        # Calculate mean and std
        if "auc" in metric:
            cols = [
                "model",
                "dataset",
                "metric",
                "train",
                "valid",
                "test",
                "suc_80",
                "suc_85",
                "suc_90",
                "suc_95",
            ]
        else:
            cols = [
                "model",
                "dataset",
                "metric",
                "train",
                "valid",
                "test",
                "mae_1",
                "mae_2",
                "mae_3",
                "mae_4",
                "mae_5",
                "mae_6",
            ]
        df_mean = df[cols].groupby(["model", "dataset", "metric"]).mean().round(4)
        df_std = df[cols].groupby(["model", "dataset", "metric"]).std().round(4)

        df_mean = df_mean.reset_index()
        df_std = df_std.reset_index()
        df_summary = df_mean[["model", "dataset", "metric"]].copy()
        if "auc" in metric:
            for col in [
                "train",
                "valid",
                "test",
                "suc_80",
                "suc_85",
                "suc_90",
                "suc_95",
            ]:
                df_summary[col] = (
                    df_mean[col].astype(str) + "±" + df_std[col].astype(str)
                )
        else:
            for col in [
                "train",
                "valid",
                "test",
                "mae_1",
                "mae_2",
                "mae_3",
                "mae_4",
                "mae_5",
                "mae_6",
            ]:
                df_summary[col] = (
                    df_mean[col].astype(str) + "±" + df_std[col].astype(str)
                )

        summary_all = f"output/{args.dataset}/summary_all.csv"
        if os.path.exists(summary_all):
            df_summary.to_csv(summary_all, mode="a", header=False, index=False)
        else:
            df_summary.to_csv(summary_all, index=False)
        print(df_summary)
