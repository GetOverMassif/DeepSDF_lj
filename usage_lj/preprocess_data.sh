# # preprocess sofa
# python preprocess_data.py \
# --data_dir /media/lj/TOSHIBA/dataset/ShapeNet/data \
# --source /media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2/ \
# --name ShapeNetV2 \
# --split examples/splits/sv2_sofas_train.json \
# --skip

# preprocess chair
python preprocess_data.py \
--data_dir /media/lj/TOSHIBA/dataset/ShapeNet/data \
--source /media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2/ \
--name ShapeNetV2 \
--split examples/splits/sv2_chairs_train_little.json \
--skip