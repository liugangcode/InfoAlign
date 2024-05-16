import os
import torch
import pickle
import logging
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset

from .data_utils import create_nx_graph, from_networkx

# from data_utils import create_nx_graph, from_networkx

logger = logging.getLogger(__name__)

class PretrainContextDataset(InMemoryDataset):
    def __init__(
        self, name="pretrain", root="raw_data", transform=None, pre_transform=None
    ):
        """
        - name (str): name of the pretraining dataset: pretrain
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        """
        self.name = name
        self.dir_name = "_".join(name.split("-"))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))
        self.threshold = pre_transform
        self.nxg_name = f"nxg_mint{pre_transform}"

        super(PretrainContextDataset, self).__init__(self.processed_root, None, None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def processed_file_names(self):
        if self.threshold is not None:
            return [f"context_{self.nxg_name}.pt"]
        else:
            return ["context_data_processed.pt"]

    def process(self):
        threshold = self.threshold if self.threshold is not None else 0.6
        folder = osp.join(self.root, "raw")
        nxg_name = f"{folder}/{self.nxg_name}.pickle"
        if not os.path.exists(nxg_name):
            G = create_nx_graph(folder, min_thres=threshold, top_compound_gene_express=0.01)
            with open(nxg_name, "wb") as f:
                pickle.dump(G, f)
        else:
            G = pd.read_pickle(nxg_name)
        
        pyg_graph = from_networkx(G)
        torch.save(self.collate([pyg_graph]), self.processed_paths[0])


# main file
if __name__ == "__main__":
    pass