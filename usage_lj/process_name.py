import os


txt_name = f"/home/lj/Documents/DeepSDF_lj/examples/splits/sv2_chairs_train_all.txt"
model_dir = f"/media/lj/TOSHIBA/dataset/ShapeNet/normal_data/SdfSamples/ShapeNetV2/03001627"
save_txt = f"/home/lj/Documents/DeepSDF_lj/examples/splits/sv2_chairs_train_all_processed.txt"

f1 = open(txt_name, 'r')
f2 = open(save_txt, 'w')

suffix = '.npz'

for line in f1.readlines():
    line = line.strip('\n')
    model_name = line.split('"')[1] + suffix
    model_path = os.path.join(model_dir, model_name)
    if os.path.exists(model_path):
        f2.write(line + '\n')
    else:
        print(model_name)

f1.close()
f2.close()