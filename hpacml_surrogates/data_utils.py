import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import numpy as np


class HDF5DataSet(Dataset):
    def __init__(self, file_path, group_name, ipt_dataset, opt_dataset, train_end_index = np.inf, target_type=None, load_entire = False):
        self.open_file = h5py.File(file_path, 'r')
        self.file_path = file_path
        self.target_type = target_type
        self.train_end_index = train_end_index
        self.load_entire = load_entire

        self.ipt_dataset, self.opt_dataset = self.get_datasets(group_name, ipt_dataset, opt_dataset)
        if target_type is None:
            self.target_dtype = torch.from_numpy(np.array([], dtype=self.opt_dataset.dtype)).dtype
        else:
            self.target_dtype = target_type
        assert len(self.ipt_dataset) == len(self.opt_dataset)
        self.length = len(self.ipt_dataset)
        if load_entire:
            tgt = self.target_type
            self.ipt_dataset = torch.tensor(self.ipt_dataset).to(tgt)
            self.opt_dataset = torch.tensor(self.opt_dataset).to(tgt)

    def get_datasets(self, group_name, ipt_dataset, opt_dataset):
        group = self.open_file[group_name]
        ipt_dataset = group[ipt_dataset]
        opt_dataset = group[opt_dataset]
        if (ipt_dataset.shape[0] == 1):
            print(f"WARNING: Found left dimension of 1 in shape {ipt_dataset.shape},"
                  f" assuming this is not necessary and removing it."
                  f" Reshaping to {ipt_dataset.shape[1:]}"
                  )
            ipt_dataset = ipt_dataset[0]
            opt_dataset = opt_dataset[0]

        if self.train_end_index == np.inf:
            self.train_end_index = ipt_dataset.shape[0]
        ipt_dataset = ipt_dataset[:self.train_end_index]
        opt_dataset = opt_dataset[:self.train_end_index]
        return ipt_dataset, opt_dataset

    def set_for_predicting_multiple_instances(self, n_instances):
        ipt_shape = self.ipt_dataset.shape
        opt_shape = self.opt_dataset.shape
        assert len(ipt_shape) == 2, "Input dataset must have 2 dimensions for compatibility with multiple instances"

        # Calculate the maximum number of full instances that can be formed
        max_full_ipt_instances = (ipt_shape[0] // n_instances) * n_instances
        max_full_opt_instances = (opt_shape[0] // n_instances) * n_instances

        # Slice the dataset to include only complete groups
        self.ipt_dataset = self.ipt_dataset[:max_full_ipt_instances].reshape((-1, ipt_shape[1] * n_instances))
        self.opt_dataset = self.opt_dataset[:max_full_opt_instances].reshape((-1, opt_shape[1] * n_instances))

        self.length = self.ipt_dataset.shape[0]

    def input_as_torch_tensor(self):
        if self.load_entire:
            return self.ipt_dataset.clone().detach()
        else:
            return torch.tensor(self.ipt_dataset).to(self.target_type)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.load_entire:
            return (self.ipt_dataset[idx].clone().detach().to(self.target_type),
                    self.opt_dataset[idx].clone().detach().to(self.target_type)
                )
        else:
            return (torch.tensor(self.ipt_dataset[idx]).to(self.target_type),
                    torch.tensor(self.opt_dataset[idx]).to(self.target_type))


class GeneratorHDF5DataSet(Dataset):
    def __init__(self, generation_command,
                 target_path, group_name,
                 ipt_dataset, opt_dataset, target_type=None
                 ):
        generation_command()
        super().__init__(target_path, group_name, 
                         ipt_dataset, opt_dataset, target_type
                         )