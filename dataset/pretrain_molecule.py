import os
import torch
import logging
import pandas as pd
import os.path as osp
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split

from .data_utils import read_graph_list

logger = logging.getLogger(__name__)

class PretrainMoleculeDataset(InMemoryDataset):
    def __init__(self, name='pretrain', root ='raw_data', transform=None, pre_transform = None):
        '''
            - name (str): name of the pretraining dataset: pretrain_all
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects
        ''' 
        self.name = name
        self.dir_name = '_'.join(name.split('-'))
        self.original_root = root
        self.root = osp.join(root, self.dir_name)
        self.processed_root = osp.join(osp.abspath(self.root))

        self.num_tasks = 1
        self.eval_metric = 'customize'
        self.task_type = 'pretrain'
        self.__num_classes__ = '-1'
        self.binary = 'False'

        super(PretrainMoleculeDataset, self).__init__(self.processed_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_data_len = self.__len__()

    def get_idx_split(self):
        full_idx = list(range(self.total_data_len))
        train_idx, valid_idx, test_idx = full_idx, [], []
        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}

    @property
    def processed_file_names(self):
        return ['mol_data_processed.pt']

    def process(self):

        mol_data_path = osp.join(self.root, 'raw', 'structure.csv.gz')
        print('Processing molecule data at folder: ' , mol_data_path)

        mol_df = pd.read_csv(mol_data_path, compression='gzip')
        mol_df = mol_df.drop_duplicates(subset="mol_id")
        data_list = read_graph_list(mol_df, keep_id=True)

        self.total_data_len = len(data_list)

        print('Pretrain molecule data loading finished with length ', self.total_data_len)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




# main file
if __name__ == "__main__":
    pass