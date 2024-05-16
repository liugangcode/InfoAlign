import ast
import os
import os.path as osp

import pandas as pd
import numpy as np
import torch

from torch_geometric.data import InMemoryDataset, Data
from .data_utils import smiles2graph, scaffold_split
# from data_utils import smiles2graph, scaffold_split

class PygRetrievalMoleculeDataset(InMemoryDataset):
    def __init__(
        self, name="chembl2k", root="raw_data", transform=None, pre_transform=None
    ):
        self.name = name
        self.root = osp.join(root, name)
        self.task_type = 'ranking'

        self.eval_metric = "rank"
        if name == "chembl2k":
            self.num_tasks = 41
            self.start_column = 4
            self.target_file = 'CP-JUMP_feature.npz'
            self.mol_file = 'CP-JUMP.csv.gz'
        elif name == "broad6k":
            self.num_tasks = 32
            self.start_column = 2
            self.target_file = 'CP-Bray_feature.npz'
            self.mol_file = 'CP-Bray.csv.gz'
        else:
            raise ValueError("Invalid dataset name")
        
        super(PygRetrievalMoleculeDataset, self).__init__(
            self.root, transform, pre_transform
        )
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.target_file, self.mol_file]

    @property
    def processed_file_names(self):
        return [f"ret_processed_{self.name}_pyg.pt"]

    def download(self):
        assert os.path.exists(
            osp.join(self.raw_dir, self.target_file)
        ), f" {osp.join(self.raw_dir, self.target_file)} does not exist"
        assert os.path.exists(
            osp.join(self.raw_dir, self.mol_file)
        ), f" {osp.join(self.raw_dir, self.mol_file)} does not exist"

    def process(self):
        target_reps = np.load(osp.join(self.raw_dir, self.target_file))['data']
        mol_df = pd.read_csv(osp.join(self.raw_dir, self.mol_file))

        if self.name == 'broad6k':
            for i in range(target_reps.shape[1]):
                mol_df['feature_'+str(i)] = target_reps[:, i]
            smiles_dict = mol_df.drop_duplicates('inchikey').set_index('inchikey')['smiles'].to_dict()
            feature_cols = [col for col in mol_df.columns if 'feature_' in col]
            df_subset = mol_df[['inchikey'] + feature_cols]
            df_subset = df_subset.groupby('inchikey').median().reset_index()
            df_subset['smiles'] = df_subset['inchikey'].map(smiles_dict)

            target_reps = df_subset[feature_cols].values
            mol_df = df_subset.drop(columns=feature_cols)
            
        mol_smiles_list = mol_df['smiles'].tolist()
        target_reps = torch.tensor(target_reps, dtype=torch.float32)

        pyg_graph_list = []
        for idx, smiles in enumerate(mol_smiles_list):
            graph = smiles2graph(smiles)

            g = Data()
            g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            try:
                g.fp = torch.tensor(graph["fp"], dtype=torch.int8).view(1, -1)
                del graph["fp"]
            except:
                pass

            g.target = target_reps[idx].view(1, -1)

            pyg_graph_list.append(g)

        print("Saving...")
        torch.save(self.collate(pyg_graph_list), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class RetrievalMoleculeDataset(object):
    def __init__(self, name="chembl2k", root="raw_data", transform="smiles"):

        assert transform in [
            "smiles",
            "fingerprint",
            "pyg",
        ], "Invalid transform type"
        
        self.name = name
        self.folder = osp.join(root, name)
        self.transform = transform

        self.eval_metric = "rank"
        if name == "chembl2k":
            self.num_tasks = 41
            self.start_column = 4
            target_file = 'CP-JUMP_feature.npz'
            mol_file = 'CP-JUMP.csv.gz'
        elif name == "broad6k":
            self.num_tasks = 32
            self.start_column = 2
            target_file = 'CP-Bray_feature.npz'
            mol_file = 'CP-Bray.csv.gz'
        else:
            raise ValueError("Invalid dataset name")
        
        self.target_file = os.path.join(self.folder, "raw", target_file)
        self.mol_file = os.path.join(self.folder, "raw", mol_file)
            
        super(RetrievalMoleculeDataset, self).__init__()
        self.prepare_data()

    def prepare_data(self):
        processed_dir = osp.join(self.folder, "processed")
        processed_file = osp.join(processed_dir, f"ret_processed_{self.name}_{self.transform}.pt")
        if not osp.exists(processed_file):
            assert os.path.exists(
                self.target_file
            ), f" {self.target_file} does not exist"
            target_reps = np.load(self.target_file)['data']
            assert os.path.exists(
                self.mol_file
            ), f" {self.mol_file} does not exist"
            mol_df = pd.read_csv(self.mol_file)

            if self.name == 'broad6k':
                for i in range(target_reps.shape[1]):
                    mol_df['feature_'+str(i)] = target_reps[:, i]
                smiles_dict = mol_df.drop_duplicates('inchikey').set_index('inchikey')['smiles'].to_dict()
                feature_cols = [col for col in mol_df.columns if 'feature_' in col]
                df_subset = mol_df[['inchikey'] + feature_cols]
                df_subset = df_subset.groupby('inchikey').median().reset_index()
                df_subset['smiles'] = df_subset['inchikey'].map(smiles_dict)

                target_reps = df_subset[feature_cols].values
                mol_df = df_subset.drop(columns=feature_cols)
                
            mol_smiles_list = mol_df['smiles'].tolist()
            inchikey_list = mol_df['inchikey'].tolist()
            target_reps = torch.tensor(target_reps, dtype=torch.float32)

            if self.transform == "smiles":
                self.data = mol_smiles_list
                self.target = target_reps
                self.inchikey_list = inchikey_list
                torch.save((mol_smiles_list, target_reps, inchikey_list), processed_file)
            elif self.transform == "fingerprint":
                from rdkit import Chem
                from rdkit.Chem import AllChem
                x_list = []
                for smiles in mol_smiles_list:
                    mol = Chem.MolFromSmiles(smiles)
                    x = torch.tensor(
                        list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)),
                        dtype=torch.float32,
                    )
                    x_list.append(x)
                x_list = torch.stack(x_list, dim=0)
                self.data = x_list
                self.target = target_reps
                self.inchikey_list = inchikey_list
                torch.save((x_list, target_reps, inchikey_list), processed_file)
            else:
                raise ValueError("Invalid transform type")
        else:
            self.data, self.target, self.inchikey_list = torch.load(processed_file)

    def __getitem__(self, idx):
        """Get datapoint(s) with index(indices)"""

        if isinstance(idx, slice):
            return self.data[idx], self.target[idx], self.inchikey_list[idx]
        
        raise IndexError("Not supported index {}.".format(type(idx).__name__))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


if __name__ == "__main__":
    pass