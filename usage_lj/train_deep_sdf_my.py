#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time

import deep_sdf
import deep_sdf.workspace as ws


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):

        return self.initial * (self.factor ** (epoch // self.interval))


class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length


def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    schedules = []

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedules.append(
                StepLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Interval"],
                    schedule_specs["Factor"],
                )
            )
        elif schedule_specs["Type"] == "Warmup":
            schedules.append(
                WarmupLearningRateSchedule(
                    schedule_specs["Initial"],
                    schedule_specs["Final"],
                    schedule_specs["Length"],
                )
            )
        elif schedule_specs["Type"] == "Constant":
            schedules.append(ConstantLearningRateSchedule(schedule_specs["Value"]))

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedules


def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = ws.get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )


def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = ws.get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )


def load_optimizer(experiment_directory, filename, optimizer):

    full_filename = os.path.join(
        ws.get_optimizer_params_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception(
            'optimizer state dict "{}" does not exist'.format(full_filename)
        )

    data = torch.load(full_filename)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = ws.get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )


# TODO: duplicated in workspace
def load_latent_vectors(experiment_directory, filename, lat_vecs):

    full_filename = os.path.join(
        ws.get_latent_codes_dir(experiment_directory), filename
    )

    if not os.path.isfile(full_filename):
        raise Exception('latent state file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    if isinstance(data["latent_codes"], torch.Tensor):
        # for backwards compatibility
        if not lat_vecs.num_embeddings == data["latent_codes"].size()[0]:
            raise Exception(
                "num latent codes mismatched: {} vs {}".format(
                    lat_vecs.num_embeddings, data["latent_codes"].size()[0]
                )
            )

        if not lat_vecs.embedding_dim == data["latent_codes"].size()[2]:
            raise Exception("latent code dimensionality mismatch")

        for i, lat_vec in enumerate(data["latent_codes"]):
            lat_vecs.weight.data[i, :] = lat_vec

    else:
        lat_vecs.load_state_dict(data["latent_codes"])

    return data["epoch"]


def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch,
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, ws.logs_filename),
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, ws.logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception('log file "{}" does not exist'.format(full_filename))

    data = torch.load(full_filename)

    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


def clip_logs(loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, epoch):

    iters_per_epoch = len(loss_log) // len(lr_log)

    loss_log = loss_log[: (iters_per_epoch * epoch)]
    lr_log = lr_log[:epoch]
    timing_log = timing_log[:epoch]
    lat_mag_log = lat_mag_log[:epoch]
    for n in param_mag_log:
        param_mag_log[n] = param_mag_log[n][:epoch]

    return (loss_log, lr_log, timing_log, lat_mag_log, param_mag_log)


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def main_function(experiment_directory, continue_from, batch_split):

    logging.debug("running " + experiment_directory)

    """ 1. 加载实验参数json文件、数据源、训练集 """
    specs = ws.load_experiment_specifications(experiment_directory)
    logging.info("Experiment description: \n" + specs["Description"][0])
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    """ 2. 导入模型，获取编码长度、检测点(周期数)、学习参数、梯度参数 """
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    logging.debug(specs["NetworkSpecs"])
    latent_size = specs["CodeLength"]
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )
    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()
    lr_schedules = get_learning_rate_schedules(specs)
    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    """ 3. 定义存储函数、终止函数、学习率调整函数、统计函数 """
    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):
        save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var
    signal.signal(signal.SIGINT, signal_handler)

    """ 4. 获取各场景点数、各场景批数、倒角距离、编码正则化、Lambda、CodeBound 等参数 """
    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True
    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    code_bound = get_spec_with_default(specs, "CodeBound", None)

    """ 5. 创建解码器，在模块级实现数据并行; 读取训练周期数、日志频率等信息 """
    # latent_size += 3
    decoder = arch.Decoder(latent_size + 3, **specs["NetworkSpecs"]).cuda()
    # decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))
    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)
    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    """ 6. 指定训练集数据划分文件，设定加载线程数，加载数据 """
    with open(train_split_file, "r") as f:
        train_split = json.load(f)
    
    # ratio_per_vec = specs["RatioPerScene"]

    # SDFSamples(数据源、训练集划分、各场景采样数、是否加载ram)
    # sdf_dataset = deep_sdf.data.SDFSamples(
    #     data_source, train_split, num_samp_per_scene, load_ram=False
    # )

    sdf_dataset = deep_sdf.data_util.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=False
    )

    scale_vecs = sdf_dataset.scale_vecs

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    # 数据加载器。结合数据集和采样器，并提供数据集的迭代
    # 支持具有单进程或多进程加载、自定义加载顺序和可选的自动批处理（整理）和内存固定的映射样式和可迭代样式数据集。
    sdf_loader = data_utils.DataLoader(
        sdf_dataset,  # 加载数据的数据集来源
        batch_size=scene_per_batch,  # 每批量数量
        shuffle=True,  # 是否需要在每个周期重新调整数据
        num_workers=num_data_loader_threads,  # 多少个子进程用于数据加载
        drop_last=True,  # 如果数据集大小不能被批次数量整除，是否删除最后的不完整批次
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))
    num_scenes = int(len(sdf_dataset) / ratio_per_vec)
    logging.info("There are {} scenes".format(num_scenes))
    logging.debug(decoder)

    """ 7. 获取符合高斯分布的随机编码 """
    # 获取满足高斯分布的编码(总模型数，编码维度，编码范数)
    # Embedding : 一个简单的查找表，用于存储固定字典和大小的嵌入

    # vecs : N * 64 , N * k * 3 -> kN * 67  (k = 8)
    print("type(num_scenes) = ", type(num_scenes))
    lat_vecs_origin = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)

    # scale_vecs = torch.tensor([[0.6,0.6,0.6]])
    # scale_vecs = scale_vecs.expand(num_scenes * ratio_per_vec, -1)

    torch.nn.init.normal_(
        lat_vecs_origin.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    lat_vecs = torch.nn.Embedding(num_scenes * ratio_per_vec, latent_size + 3, max_norm=code_bound)

    # print("shape0 = ", lat_vecs_origin.weight.data.shape)
    # print("shape1 = ", lat_vecs_origin.weight.data.unsqueeze(1).expand(-1,ratio_per_vec,-1).reshape(-1,latent_size).shape)
    # print("shape2 = ", scale_vecs.shape)
    
    lat_vecs.weight.data = \
        torch.cat((lat_vecs_origin.weight.data.unsqueeze(1).expand(-1,ratio_per_vec,-1).reshape(-1,latent_size),
                  scale_vecs), 1)

    # print("lat_vecs.weight.data = \n", lat_vecs.weight.data)

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs_origin)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:

        logging.info('continuing from "{}"'.format(continue_from))

        lat_epoch = load_latent_vectors(
            experiment_directory, continue_from + ".pth", lat_vecs
        )

        model_epoch = ws.load_model_parameters(
            experiment_directory, continue_from, decoder
        )

        optimizer_epoch = load_optimizer(
            experiment_directory, continue_from + ".pth", optimizer_all
        )

        loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = load_logs(
            experiment_directory
        )

        if not log_epoch == model_epoch:
            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = clip_logs(
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
            )

        if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
            raise RuntimeError(
                "epoch mismatch: {} vs {} vs {} vs {}".format(
                    model_epoch, optimizer_epoch, lat_epoch, log_epoch
                )
            )

        start_epoch = model_epoch + 1

        logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    # decoder parameters : (1843579)
    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    # shape code parameters : (20588 = 3217 codes * 64 dim)
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    # Training cycle
    for epoch in range(start_epoch, num_epochs + 1):
        
        start = time.time()
        logging.info("epoch {}...".format(epoch))

        print("lat_vecs.weight.data = \n", lat_vecs.weight.data)

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        cnt_sdf = 0
        # 根据模型遍历sdf数据, 遍历 cnt_sdf = 0～133, 134 * 24 = 3216
        # 需要处理的数据： 3217 * 16384 * (64 + 3 + 1)
        #            -> 3217 * 8 * 16384 * (64 + 3 + 3 + 1)
        for sdf_data, indices in sdf_loader:
            # print("[cnt_sdf : %d]" % cnt_sdf)
            cnt_sdf += 1
            
            # sdf_data: 
            # torch.Size([24, 16384, 4]) -> torch.Size([393216, 4])

            # print("sdf_data = ", sdf_data)
            # print("sdf_data.shape = ", sdf_data.shape)
            # Process the input data: x,y,z,sdf
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]
            sdf_data.requires_grad = False
            xyz = sdf_data[:, 0:3]

            # sdf_gt: torch.Size([393216, 1])
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)
            
            # xyz: <class 'tuple'>
            xyz = torch.chunk(xyz, batch_split)

            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss = 0.0
            optimizer_all.zero_grad()

            # 分批放入解码器训练，如分4批进行, 98304 = 24 * 16384 / 4
            for i in range(batch_split):
                # print("[batch_split : %d]" % i)

                batch_vecs = lat_vecs(indices[i])

                # batch_vecs:  torch.Size([98304, 64])
                #             tensor([[code1 ......],
                #                     [code2 ......],
                #                         ......
                #                     [code_n .....]])

                # lat_vecs: Embedding(3217, 64, max_norm=1.0)

                # indices[i] : torch.Size([98304])
                #              tensor([2174, 2174, 2174,  ..., 2354, 2354, 2354])

                scale = torch.tensor([0.6, 0.6, 0.6])
                scale = scale.expand(indices[i].shape[0],-1)

                # scale: torch.Size([98304, 3])
                # xyz[i]: torch.Size([98304, 3])
                # print("\nlat_vecs", lat_vecs)
                # print("\nindices[i]", indices[i])
                # print("\nindices[i].shape", indices[i].shape)
                # print("\nscale.shape : ", scale.shape)
                # print("\nxyz[i].shape : ", xyz[i].shape)
                # print("\nbatch_vecs : ", batch_vecs)
                # print("\nscale : ", scale)
                # print("\nxyz[i] : ", xyz[i])
                
                # 合成输入
                # input = torch.cat([batch_vecs, scale, xyz[i]], dim=1)
                input = torch.cat([batch_vecs, xyz[i]], dim=1)
                # print("input size = ", input.shape)
                # input = torch.cat([batch_vecs, xyz[i]], dim=1)

                # NN optimization， 将输入放入decoder中
                pred_sdf = decoder(input)

                # 裁剪掉最大最小范围外的值
                if enforce_minmax:
                    pred_sdf = torch.clamp(pred_sdf, minT, maxT)

                # 计算平均损失
                chunk_loss = loss_l1(pred_sdf, sdf_gt[i].cuda()) / num_sdf_samples

                # 如果要进行编码正则化
                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = (
                        code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                    ) / num_sdf_samples
                    chunk_loss = chunk_loss + reg_loss.cuda()

                # 损失的反向传播，并且添加到块损失中
                chunk_loss.backward()
                batch_loss += chunk_loss.item()

            logging.debug("loss = {}".format(batch_loss))
            loss_log.append(batch_loss)

            # 梯度范数
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            optimizer_all.step()

        """ 计算用时, 写入日志，保存文件 """
        end = time.time()
        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)
        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])
        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))
        append_parameter_magnitudes(param_mag_log, decoder)

        # save checkpoint's Parameters
        if epoch in checkpoints:
            save_checkpoints(epoch)

        # save latest Parameters and Log
        if epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.continue_from, int(args.batch_split))
