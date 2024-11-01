import csv
import random
import os
from torch.utils.data import Dataset
import numpy as np
from nilearn import image
import torch
import nibabel as nib
import math


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        csv_list = list(reader)
    filenames = [a[0] for a in csv_list[1:]]
    labels = [a[1] for a in csv_list[1:]]
    return filenames, labels


def data_split(train_ratio, valid_ratio, label_csv, repe_time):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f)
        list = list(reader)
    titles, data = list[0:1], list[1:]
    # Shuffle the entire dataset
    random.seed(0)
    random.shuffle(data)
    # Calculate the number of samples for each split
    total_samples = len(data)
    train_size = int(train_ratio * total_samples)
    valid_size = int(valid_ratio * total_samples)
    for i in range(repe_time):
        # Split the dataset into training, validation, and test sets
        train_data = data[:train_size]
        valid_data = data[train_size:train_size + valid_size]
        test_data = data[train_size + valid_size:]
        folder = 'exp{}/'.format(i)
        if not os.path.exists(folder):
            os.mkdir(folder)

        # Write the split datasets to CSV files
        with open(folder + 'train.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(titles + train_data)

        with open(folder + 'valid.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(titles + valid_data)

        with open(folder + 'test.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(titles + test_data)


def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan


def nifti_to_numpy(file):
    data = image.load_img(file)
    data = data.get_fdata()
    return data


class Load_data(Dataset):
    def __init__(self, Data_dir, exp_idx, stage):
        self.Data_dir = Data_dir
        self.Data_list, self.Label_list = read_csv('./exp{}/{}.csv'.format(exp_idx, stage))

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        data = nifti_to_numpy(self.Data_dir + self.Data_list[idx] + '.nii')  # nii to npy

        # Convert to float
        data = torch.FloatTensor(data)
        # Add a dimension
        data = data.unsqueeze(0)
        return data, label


def data_sample(label_csv, sample_size, output_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    titles, data = your_list[0:1], your_list[1:]
    # Shuffle the entire dataset
    random.seed(0)
    random.shuffle(data)
    # Calculate the number of samples for each split
    total_samples = len(data)
    train_size = int(sample_size)
    train_data = data[:train_size]

    # Write the split datasets to CSV files
    with open(output_csv, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(titles + train_data)


class AAL_process(Dataset):
    def __init__(self, Data_dir, exp_idx, stage, AAL, AAL_path):
        self.Data_dir = Data_dir
        self.AAL = AAL
        self.AAL_path = AAL_path
        self.Data_list, self.Label_list = read_csv('./exp{}/{}.csv'.format(exp_idx, stage))

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        aal_template = nib.load(self.AAL_path)
        aal_data = aal_template.get_fdata()
        mask = np.isin(aal_data, self.AAL)
        inverted_mask = np.logical_not(mask)
        data = nib.load(self.Data_dir + self.Data_list[idx] + '.nii')
        masked_data = np.where(inverted_mask, data.get_fdata(), 0)
        # Convert to float
        data = torch.FloatTensor(masked_data)
        # Add a dimension
        data = data.unsqueeze(0)
        return data, label


def shapley(data, factor, sequence):
    num_tests = len(data[0]['test'])
    Shapley_values = [0] * num_tests
    for i in range(sequence):
        S = math.factorial(i) * math.factorial(sequence - i - 1) / math.factorial(sequence)

        # Select combinations that contain 'factor' and have a length of i+1.
        VSj = [item for item in data if
               factor in item['brain_region_combination'] and len(item['brain_region_combination']) == i + 1]

        marginal_scores_sum = [0] * num_tests

        # Iterate over the selected combinations
        for VSj_dict in VSj:
            other_regions = [region for region in VSj_dict['brain_region_combination'] if region != factor]

            # Identify combinations that do not contain 'factor' but include the same other elements
            for item in data:
                if factor not in item['brain_region_combination'] and all(
                        region in item['brain_region_combination'] for region in other_regions) and len(
                    item['brain_region_combination']) == i:
                    marginal_scores = [S * (VSj_dict['test'][k] - item['test'][k]) for k in range(num_tests)]
                    marginal_scores_sum = [marginal_scores_sum[k] + marginal_scores[k] for k in range(num_tests)]

        # Multiply the cumulative marginal scores by S and save the result in Shapley_values
        Shapley_values = [Shapley_values[k] + marginal_scores_sum[k] for k in range(num_tests)]

    return Shapley_values
