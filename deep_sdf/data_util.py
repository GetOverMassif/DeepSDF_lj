#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import deep_sdf.workspace as ws


def get_instance_filenames(data_source, split, ratio_per_vec):
    npzfiles, txtfiles = [], []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_path = os.path.join(
                    dataset, class_name, instance_name
                )
                for i in range(ratio_per_vec):
                    instance_filename = os.path.join(instance_path, str(i) + ".npz")
                    instance_txtname = os.path.join(instance_path, str(i) + ".txt")
                    # print("[path]", os.path.join(data_source, ws.sdf_samples_subdir, instance_filename))
                    if not os.path.isfile(
                        os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                    ) or not os.path.isfile(
                        os.path.join(data_source, ws.sdf_samples_subdir, instance_txtname)
                    ):
                        # raise RuntimeError(
                        #     'Requested non-existent file "' + instance_filename + "'"
                        # )
                        logging.warning(
                            "Requested non-existent file '{}' or '{}'".format(instance_filename, instance_txtname)
                        )
                    npzfiles += [instance_filename]
                    txtfiles += [os.path.join(data_source, ws.sdf_samples_subdir, instance_txtname)]
    return npzfiles, txtfiles

def get_instance_scales(txtfiles):
    files_num = len(txtfiles)
    scales_np = np.zeros((files_num, 3), dtype = np.float32)
    for i in range(files_num):
        scales_np[i] = np.loadtxt(txtfiles[i], dtype = np.float32)[:3]
    return torch.from_numpy(scales_np)

class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]

def find_meshes_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    # elif len(mesh_filenames) > 1:
    #     raise MultipleMeshFileError()
    return mesh_filenames


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def read_sdf_samples_into_ram(filename):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])

    return [pos_tensor, neg_tensor]


def unpack_sdf_samples(filename, txtname, subsample = None):
    # npz: <class 'numpy.lib.npyio.NpzFile'>
    npz = np.load(filename)
    # print("npz.type", type(npz))
    scale_np = np.loadtxt(txtname, dtype=np.float32)

    if subsample is None:
        return npz
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


def unpack_sdf_samples_from_ram(data, subsample=None):
    if subsample is None:
        return data
    pos_tensor = data[0]
    neg_tensor = data[1]

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    pos_start_ind = random.randint(0, pos_size - half)
    sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    return samples


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,  # 
        data_source,  # 数据源
        split,  # 划分集
        ratio_per_vec, 
        subsample,  # 每个场景的采样点数量
        load_ram = False,
        print_filename = False,
        num_files=1000000,
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles, self.txtfiles = get_instance_filenames(data_source, split, ratio_per_vec)

        self.scale_vecs = get_instance_scales(self.txtfiles)

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        # 不加载
        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
                self.loaded_data.append(
                    [
                        pos_tensor[torch.randperm(pos_tensor.shape[0])],
                        neg_tensor[torch.randperm(neg_tensor.shape[0])],
                    ]
                )

    def __len__(self):
        # print("measure len = ", len(self.npyfiles))
        return len(self.npyfiles)

    def __getitem__(self, idx):
        # print("get item idx = %d" % idx)
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        txtname = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.txtfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample),
                idx,
            )
        else:
            # print("before unpack_sdf_samples")
            return unpack_sdf_samples(filename, txtname, self.subsample), idx
