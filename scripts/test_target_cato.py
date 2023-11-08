import os
import os.path as osp
import subprocess

# target_catogory=$1

DEEPSDF = f"/home/lj/Documents/DeepSDF_lj"
SHAPENET = f"/media/lj/TOSHIBA/dataset/ShapeNet"
DATA_DIR = f"/media/lj/TOSHIBA/dataset/ShapeNet/data"

CLASS = f"bottles"

TEST_DATA = f"/media/lj/TOSHIBA/dataset/ShapeNet/test_data"

# pre-process the sofas training set (SDF samples)
# 这一步会在 $SHAPENET/$DATA_DIR下 生成 三个子文件夹：
#               NormalizationParameters,SdfSamples,SurfaceSamples
# 分别存储的是：法方向信息(npz文件),SDF采样数据(npz文件),表面采样点信息(ply文件)
# 指令的其他参数：--threads 默认为8
# 用时：48个instance，用了6.93s

process_filename = f"{DEEPSDF}/usage/preprocess_data.py"
train_filename = f"{DEEPSDF}/usage/train_deep_sdf.py"
reconstruct_filename = f"{DEEPSDF}/usage/reconstruct.py"

cmd = f"python {process_filename} \
    --data_dir {TEST_DATA} \
    --source {SHAPENET}/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_{CLASS}_test.json \
    --test --skip"

p = subprocess.Popen(cmd, shell=True)
return_code = p.wait()

cmd = f"python {process_filename} \
    --data_dir {TEST_DATA} \
    --source {SHAPENET}/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_{CLASS}_test.json \
    --surface --skip"
p = subprocess.Popen(cmd, shell=True)
return_code = p.wait()

# reconstruct
cmd = f"python {reconstruct_filename} \
    -e /media/lj/TOSHIBA/dataset/DeepSDF/{CLASS}_64 \
    -c 2000 \
    --split examples/splits/sv2_{CLASS}_test.json \
    -d {TEST_DATA} \
    --skip"
p = subprocess.Popen(cmd, shell=True)
return_code = p.wait()

