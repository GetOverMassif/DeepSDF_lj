
# # navigate to the DeepSdf root directory
# cd [...]/DeepSdf

# create a home for the data
mkdir data

DEEPSDF=/home/lj/Documents/DeepSDF_lj
SHAPENET=/media/lj/TOSHIBA/dataset/ShapeNet
DATA_DIR=/media/lj/TOSHIBA/dataset/ShapeNet/data
CLASS=chairs

# pre-process the sofas training set (SDF samples)
# 这一步会在 $SHAPENET/$DATA_DIR下 生成 三个子文件夹：
#               NormalizationParameters,SdfSamples,SurfaceSamples
# 分别存储的是：法方向信息(npz文件),SDF采样数据(npz文件),表面采样点信息(ply文件)
# 指令的其他参数：--threads 默认为8
# 用时：48个instance，用了6.93s


python $DEEPSDF/usage/preprocess_data.py --data_dir $DATA_DIR --source $SHAPENET/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_$CLASS\_train_little.json --skip

# train the model
# 这里指定了迭代优化的模型存储的位置，同时，这个文件夹下面的
# specs.json 存储了众多配置信息：
#    包括：数据源路径，训练用例文件，测试用例文件，网络架构，编码长度，
#         训练过程的周期总数，多少个周期保存一次模型，额外的保存周期数，学习率，
#         每个场景包含的采样数，每个batch包含的场景数, 数据加载的线程数等
# GPU现存占用计算方法：
# 生成解码器模型的文件大小(保持原网络结构)：64位code-7.4MB，
python $DEEPSDF/usage/train_deep_sdf.py -e $DEEPSDF/examples/$CLASS

# pre-process the sofa test set (SDF samples)
python preprocess_data.py --data_dir data --source [...]/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_sofas_test.json --test --skip

# pre-process the sofa test set (surface samples)
python preprocess_data.py --data_dir data --source [...]/ShapeNetCore.v2/ --name ShapeNetV2 --split examples/splits/sv2_sofas_test.json --surface --skip

# reconstruct meshes from the sofa test split (after 2000 epochs)
python reconstruct.py -e examples/sofas -c 2000 --split examples/splits/sv2_sofas_test.json -d data --skip

# evaluate the reconstructions
python evaluate.py -e examples/sofas -c 2000 -d data -s examples/splits/sv2_sofas_test.json