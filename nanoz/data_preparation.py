#!/usr/bin/env python3

import logging

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import nanoz.pre_processing as nzpp
import nanoz.transform as nztransform
from nanoz.modeling import AvailableAlgorithm


extract_train = [
    'data_paths_train', 'chips_train', 'gases_train', 'minibatch_size_train', 'minibatch_step_train', 'pre_processing_train',
    'transform_train']

extract_test= [
    'data_paths_test', 'chips_test', 'gases_test', 'minibatch_size_test', 'minibatch_step_test', 'pre_processing_test',
    'transform_test']



def tensor_shuffle(tensor):
    """
    Shuffle the rows of a PyTorch tensor along the first dimension.

    This function shuffles the rows of a PyTorch tensor along the first dimension, ensuring that the data remains
    consistent across dimensions.

    Parameters
    ----------
    tensor : torch.Tensor
        The input PyTorch tensor to shuffle.

    Returns
    -------
    torch.Tensor
        The shuffled tensor with rows randomly permuted.
    """
    random_idx = torch.randperm(tensor.shape[0])
    return tensor[random_idx].view(tensor.size())


def get_string_index(string_list, string):
    if isinstance(string, str):
        string = [string]
    elif isinstance(string, list):
        pass
    else:
        raise TypeError(f"Substring must be a string or a list of string."
                        f"Got {type(string)}: {string} instead.")

    logging.debug(f"Get index for {string} in {string_list}")
    index = []

    for sub in string:
        for s in string_list:
            if sub == s:
                index.append(string_list.index(s))

    index = list(dict.fromkeys(index))  # remove duplicate
    logging.debug(f"Index found: {index}")
    return index


def get_substring_index(string_list, substring):
    """
    Get the indices of strings in the given list that contain the specified substrings.

    This function searches for the specified substrings within the strings of the string list and returns a list of
    unique indices corresponding to the strings that contain the substrings.

    Parameters
    ----------
    string_list : list of str
        The list of strings to search within.

    substring : str or list of str
        The substring or substrings to search for. If a single substring is provided, it is converted into a list.

    Returns
    -------
    list
        A list of unique indices corresponding to the strings that contain the specified substrings.
    """
    if isinstance(substring, str):
        substring = [substring]
    elif isinstance(substring, list):
        pass
    else:
        raise TypeError(f"Substring must be a string or a list of string."
                        f"Got {type(substring)}: {substring} instead.")

    logging.debug(f"Get index for {substring} in {string_list}")
    index = []

    for sub in substring:
        for string in string_list:
            if sub in string:
                index.append(string_list.index(string))

    index = list(dict.fromkeys(index))  # remove duplicate
    logging.debug(f"Index found: {index}")
    return index


class DatasetFactory:
    """
    A factory class for creating different types of datasets.

    This class provides a static method `create_dataset` that returns an instance of either an `UnfoldRegressionDataset`
    or `UnfoldClassificationDataset` class, depending on the value of the `algo` argument.

    Methods
    -------
    create_dataset(algo, **kwargs)
        Create a dataset instance based on the specified algorithm name.
    """
    @staticmethod
    def create_dataset(algo,mode, **kwargs):
        """
        Create a dataset instance based on the specified algorithm name.

        Parameters
        ----------
        algo : str
            The name of the algorithm to use for creating the dataset.
            Must be one of the available algorithms for modeling.

        **kwargs : dict
            Additional keyword arguments to pass to the dataset class constructor.

        Returns
        -------
        UnfoldRegressionDataset or UnfoldClassificationDataset
            An instance of the appropriate dataset class.
        """
        algo_type = AvailableAlgorithm.get_type(algo)
        if algo_type == "regression":
            logging.debug(f"Creating Untrain_classficationfoldRegressionDataset with {kwargs}")
            return UnfoldRegressionDataset(mode,**kwargs)
        elif algo_type == "classification":
            logging.debug(f"Creating UnfoldClassificationDataset with {kwargs}")
            return UnfoldClassificationDataset(mode,**kwargs)
        elif algo_type == "autoencoder":
            logging.debug(f"Creating UnfoldAutoEncoderDataset with {kwargs}")
            return UnfoldAutoEncoderDataset(**kwargs)
        else:
            raise ValueError(f"Invalid algorithm: {algo}")


class UnfoldDataset(Dataset):
    def __init__(self, mode ,config_data, device):
        if mode=="train":
            extracted_train = {key: config_data[key] for key in extract_train}

            self.config_data = extracted_train
            self.pre_processing = self._get_pre_processing("pre_processing_train")
            self.transform = self._get_transform("transform_train")
        else :
            extracted_test = {key: config_data[key] for key in extract_test}
            self.pre_processing = self._get_pre_processing("pre_processing_test")
            self.transform = self._get_transform("transform_test")

            self.config_data = extracted_test
        self.device = device
        self.mode=mode
        self.file_size = []
     
        self.data = self.load_data(mode)
        self.classes = None

        self._available_idx = self._get_available_idx()
        self.minibatch_idx = self._available_idx
        self._x = torch.empty(0, dtype=torch.float32)
        self._y = torch.empty(0, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = {"x": self._x, "y": self._y}
        if self.transform:
            for var in self.transform:
                sample[var] = transforms.Compose(self.transform[var])(sample[var])
        return sample["x"], sample["y"]

    def __len__(self):
        return len(self._available_idx)

    @property
    def ground_truth(self):
        """
        Get the ground truth.

        Returns
        -------
        torch.Tensor
            The ground truth.
        """
        gt = torch.index_select(self._y, 0, self._available_idx)
        logging.debug(f"Getting ground truth, size: {gt.shape}")
        return gt.cpu().numpy()

    def _get_pre_processing(self, name):
        if name in self.config_data:
            step_list = []
            for step in self.config_data[name]:
                step_list.append(getattr(nzpp, step["name"])(**step))

            return step_list
        else:
            logging.debug(f"{name} not found in config file.")
            return None

    def _get_transform(self, name):
        if name in self.config_data:
            steps = {}
            for var in self.config_data[name]:
                step_list = []
                for step in self.config_data[name][var]:
                    #step["device"] = self.device  # All transformation type should have access to the device information
                    step_list.append(getattr(nztransform, step["name"])(**step))
                steps[var] = step_list

            return steps
        else:
            logging.debug(f"{name} not found in config file.")
            return None

    def _get_available_idx(self):
        batch_idx = []
        ptr_counter = 0
        if self.mode == "train":
            for _ in self.config_data["chips_train"]:
                
                for file_size in self.file_size:
                    batch_idx.extend(list(range(ptr_counter + self.config_data["minibatch_size_train"],
                                                ptr_counter + file_size,
                                                self.config_data["minibatch_step_train"])))
                    ptr_counter += file_size
        else:
            for _ in self.config_data["chips_test"]:
                for file_size in self.file_size:
                    batch_idx.extend(list(range(ptr_counter + self.config_data["minibatch_size_test"],
                                                ptr_counter + file_size,
                                                self.config_data["minibatch_step_test"])))
                    ptr_counter += file_size
        batch_idx = torch.tensor(batch_idx, dtype=torch.int64)
        logging.debug(f"Size of the available index: {batch_idx.size()}.")
        return batch_idx.to(self.device)

    def _extract_data(self):
        x, y = [], []
        columns_list = list(self.data)

        if self.mode == "train":
            y_idx = get_string_index(columns_list, self.config_data["gases_train"][0])
            data = self.data.to_numpy()

            # Execute training mode logic
            for chip in self.config_data["chips_train"]:
                logging.debug(f"Extend data from chip {chip}.")
                x_idx = get_substring_index(columns_list, chip)
                if x_idx:
                    x.extend(data[:, x_idx])
                    y.extend(data[:, y_idx])
        else:
            y_idx = get_string_index(columns_list, self.config_data["gases_test"][0])
            data = self.data.to_numpy()

            # Execute testing mode logic
            for chip in self.config_data["chips_test"]:
                logging.debug(f"Extend data from chip {chip}.")
                x_idx = get_substring_index(columns_list, chip)
                if x_idx:
                    x.extend(data[:, x_idx])
                    y.extend(data[:, y_idx])
  
        return x, y


    def shuffle(self):
        """
        Shuffle the indices of the minibatch.

        This method shuffles the indices of the minibatch used for training. It modifies the `minibatch_idx` attribute
        of the current object in-place.
        """
        self.minibatch_idx = tensor_shuffle(self.minibatch_idx)
        logging.debug(f"Shuffled minibatch indices.")

    def unshuffle(self):
        """
        Restore the original order of the indices in the minibatch.

        This method restores the original order of the indices in the minibatch. It assigns the `minibatch_idx`
        attribute of the current object with the original indices stored in the `_available_idx` attribute.
        """
        self.minibatch_idx = self._available_idx
        logging.debug(f"Unshuffled minibatch indices.")

    def load_data(self,mode):
        if mode == 'train':
            
            return self.load_and_preprocess_data(self.config_data["data_paths_train"])
        else:
            return self.load_and_preprocess_data(self.config_data["data_paths_test"])


    def load_and_preprocess_data(self, data_paths):
        data = pd.DataFrame()
        
        for data_path in data_paths:
            logging.debug("")
            logging.debug(f"Loading data from {data_path}.")
            for data_file in data_path["data"]:
                df = pd.read_csv("data/"+data_file)
                logging.debug(f"Shape of the dataframe {df.shape}.")
                with pd.option_context('display.max_columns', 40):
                    logging.debug(f"\n{df.describe(include='all')}")

                if self.pre_processing:  # TODO: add artefacts correction and nan with split
                    df = transforms.Compose(self.pre_processing)(df)
                # TODO : pre_processing
                nan_rows = df[df.isnull().any(axis=1)].index.to_list()  # TODO: if missing data on chip but not on other
                df.drop(df.index[nan_rows], inplace=True)
                logging.debug(f"Shape of the dataframe after pre-processing: {df.shape}.")
                self.file_size.append(len(df))

                if self.mode == "train":
                    if self.config_data["minibatch_size_train"] > self.file_size[-1]:
                        raise ValueError(f"Size of {data_path} is lower than the batch_size "
                                        f"parameter ({self.config_data['minibatch_size_train']}).")
                    
                    data = pd.concat([data, df], axis=0, ignore_index=True)
                else:
                    if self.config_data["minibatch_size_test"] > self.file_size[-1]:
                        raise ValueError(f"Size of {data_path} is lower than the batch_size "
                                        f"parameter ({self.config_data['minibatch_size_test']}).")
                    
                    data = pd.concat([data, df], axis=0, ignore_index=True)
                logging.debug(f"Shape of the full dataframe: {data.shape}.")
                return data

class UnfoldRegressionDataset(UnfoldDataset):
    def __init__(self, mode,config_data, device):
        super().__init__(mode,config_data, device)
        self._x, self._y = self.prepare_torch_tensor()
        
    def __getitem__(self, idx):
        idx_end = self.minibatch_idx[idx]
        if (self.mode=="train"):
            idx_start = idx_end - self.config_data["minibatch_size_train"]
        else:
            idx_start = idx_end - self.config_data["minibatch_size_test"]
    
        sample = {"x": self._x[idx_start:idx_end, :], "y": self._y[idx_end, :]}
        if self.transform:
            for var in self.transform:
                sample[var] = transforms.Compose(self.transform[var])(sample[var])
        return sample["x"], sample["y"]

    def prepare_torch_tensor(self):
        x, y = self._extract_data()
        x = torch.tensor(x, dtype=torch.float32)
        logging.debug(f"Size of features: {x.shape}.")
        y = torch.tensor(y, dtype=torch.float32)
        logging.debug(f"Size of labels: {y.shape}.")
        return x.to(self.device), y.to(self.device)


class UnfoldClassificationDataset(UnfoldDataset):
    def __init__(self, mode,config_data, device):
        super().__init__(mode,config_data, device)
        self._x, self._y = self.prepare_torch_tensor()
        self.classes = self._get_classes()

    def __getitem__(self, idx):
        idx_end = self.minibatch_idx[idx]
        if (self.mode=="train"):
            idx_start = idx_end - self.config_data["minibatch_size_train"]
        else:
            idx_start = idx_end - self.config_data["minibatch_size_test"]
    
        sample = {"x": self._x[idx_start:idx_end, :], "y": self._y[idx_end]}
        if self.transform:
            for var in self.transform:
                sample[var] = transforms.Compose(self.transform[var])(sample[var])
        return sample["x"], sample["y"]

    def _get_classes(self):
        if "pre_processing" in self.config_data:
            for step in self.config_data["pre_processing"][::-1]:  # reverse order
                if "OrdinalEncoderFromInterval" in step["name"]:  # TODO: generalize
                    return step["intervals"]
        return []

    def prepare_torch_tensor(self):
        x, y = self._extract_data()
        x = torch.tensor(x, dtype=torch.float32)
        logging.debug(f"Size of features: {x.shape}.")
        y = torch.tensor(y, dtype=torch.int64)
        y = torch.squeeze(y)  # Due to the loss function
        logging.debug(f"Size of labels: {y.shape}.")
        return x.to(self.device), y.to(self.device)


class UnfoldAutoEncoderDataset(UnfoldDataset):
    def __init__(self, config_data, device):
        super().__init__(config_data, device)
        self._x, self._y = self.prepare_torch_tensor()

    def __getitem__(self, idx):
        idx_end = self.minibatch_idx[idx]
        if (self.mode=="train"):
            idx_start = idx_end - self.config_data["minibatch_size_train"]
        else:
            idx_start = idx_end - self.config_data["minibatch_size_test"]
            sample = {"x": self._x[idx_start:idx_end, :]}
        if self.transform:
            for var in self.transform:
                sample[var] = transforms.Compose(self.transform[var])(sample[var])
        return sample["x"], sample["x"]

    def prepare_torch_tensor(self):
        x, _ = self._extract_data()
        x = torch.tensor(x, dtype=torch.float32)
        logging.debug(f"Size of features: {x.shape}.")
        logging.debug(f"Size of labels: {x.shape}.")
        return x.to(self.device), x.to(self.device)
