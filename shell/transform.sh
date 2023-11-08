DEEPSDF=/home/lj/Documents/DeepSDF_lj
SHAPENET=/media/lj/TOSHIBA/dataset/ShapeNet
DATA_DIR=/media/lj/TOSHIBA/dataset/ShapeNet/deformed_data
CLASS=chairs

TEST_DATA=/media/lj/TOSHIBA/dataset/ShapeNet/test_data


python $DEEPSDF/usage/preprocess_data.py \
    --data_dir $TEST_DATA \
    --source $SHAPENET/ShapeNetCore.v2/ \
    --name ShapeNetV2 \
    --split examples/splits/sv2_$CLASS\_train.json \
    --surface --skip
