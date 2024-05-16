import os
import pandas as pd

import torch
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
)
__all__ = ["validate", "init_weights", "AverageMeter", "save_prediction"]

def eval_func(pred, true, reduction=True):
    unique_values = np.unique(true[~np.isnan(true)])
    task_type = 'classification' if set(unique_values).issubset({0, 1}) else 'regression'

    if task_type == 'classification':
        rocauc_list = []
        count_dict = {80: [], 85: [], 90: [], 95: []}
        for i in range(true.shape[1]):
            if np.sum(true[:, i] == 1) > 0 and np.sum(true[:, i] == 0) > 0:
                is_labeled = true[:, i] == true[:, i]
                score = roc_auc_score(true[is_labeled, i], pred[is_labeled, i])
                for threshold in count_dict.keys():
                    count_dict[threshold].append(int(score >= threshold / 100))
                rocauc_list.append(score)
            else:
                for threshold in count_dict.keys():
                    count_dict[threshold].append(np.nan)

        if len(rocauc_list) == 0:
            raise RuntimeError(
                "No positively labeled data available. Cannot compute ROC-AUC."
            )
        return {"roc_auc": sum(rocauc_list) / len(rocauc_list), "count": count_dict} if reduction else {"roc_auc": rocauc_list}
    
    elif task_type == 'regression':
        mae_list = []
        for i in range(true.shape[1]):
            is_labeled = ~np.isnan(true[:, i])
            mae_score = mean_absolute_error(true[is_labeled, i], pred[is_labeled, i])
            mae_list.append(mae_score)
        return {"avg_mae": np.mean(mae_list), "mae_list": mae_list}

def save_prediction(model, device, loader, dataset, output_dir, seed):
    y_true = []
    y_pred = []
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    data_name = "_".join(dataset.name.split("-")) if "moltoxcast" in dataset.name else dataset.name
    assay_path = f"raw_data/{data_name}/raw/assays.csv.gz"
    assay_df = pd.read_csv(assay_path, compression="gzip")
    if "moltoxcast" in dataset.name:
        assay_names = assay_df.columns[:-2]
    else:
        assay_names = assay_df.columns[dataset.start_column :]
    
    y_pred[np.isnan(y_true)] = np.nan
    os.makedirs(output_dir, exist_ok=True)

    df_pred = pd.DataFrame(y_pred, columns=[name + "_pred" for name in assay_names])
    df_true = pd.DataFrame(y_true, columns=[name + "_true" for name in assay_names])
    df = pd.concat([df_pred, df_true], axis=1)
    df.to_csv(os.path.join(output_dir, f"preds-{seed}.csv"), index=False)

    returned_dict = eval_func(y_pred, y_true, reduction=False)
    results = returned_dict.get('roc_auc', None)
    if results is None:
        results = returned_dict.get('mae_list', None)
        if results is None:
            raise ValueError("Invalid task type")
    results_dict = dict(zip(assay_names, results))
    sorted_dict = dict(
        sorted(results_dict.items(), key=lambda item: item[1], reverse=True)
    )
    df_sorted = pd.DataFrame(
        list(sorted_dict.items()), columns=["Assay Name", "Result"]
    )
    df_sorted.to_csv(os.path.join(output_dir, f"result-{seed}.csv"), index=False)

def validate(args, model, loader):
    y_true = []
    y_pred = []
    device = args.device
    model.eval()
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    return eval_func(y_pred, y_true)


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == "default":
                pass
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_base(base, x):
    return np.log(x) / np.log(base)


def _eval_rocauc(y_true, y_pred):
    """
    compute ROC-AUC averaged across tasks
    """
    rocauc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(
                roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            )

    if len(rocauc_list) == 0:
        raise RuntimeError(
            "No positively labeled data available. Cannot compute ROC-AUC."
        )
    return {"rocauc": sum(rocauc_list) / len(rocauc_list)}


if __name__ == "__main__":
    pass
