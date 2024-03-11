
# 首先编译好,需要安装CLI12,nanoflann
DEEPSDF_DIR=/home/lj/Documents/codes/DeepSDF_lj
DATA_SAVE_DIR=/media/lj/TOSHIBA/dataset/ShapeNet/data
SHAPENET_DIR=/media/lj/TOSHIBA/dataset/ShapeNet

SHAPENETCORE2_DIR=/media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2

# CLASS=keyboards
# CLASS=bowls
# CLASS=laptops
CLASS=$1


# 1. 查询，把新类别和对应的shapenet编号插入大盘ShapeNetClass.json中

# 2. 修改 scripts/generate_new_example.py 中的类别
python scripts/generate_new_example.py -c $CLASS -s $SHAPENETCORE2_DIR -d $DATA_SAVE_DIR

# 3. 数据预处理
python $DEEPSDF_DIR/usage/preprocess_data.py \
    --data_dir $DATA_SAVE_DIR \
    --source $SHAPENET_DIR/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_$CLASS\_train.json \
    --skip

# 4. 训练
python $DEEPSDF_DIR/usage/train_deep_sdf.py \
    -e $DEEPSDF_DIR/examples/$CLASS

# 5. 测试
