import os
import os.path as osp
import json

model_dir = f"/media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2"
json_path_classes_shapenet = f"/home/lj/Documents/codes/DeepSDF_lj/ShapeNetClasses.json"
json_path_classes_shapenet_lack = f"/home/lj/Documents/codes/DeepSDF_lj/ShapeNetClassesLack.yml"

taxonomy_file = osp.join(model_dir, f"taxonomy.json")
taxonomy_filtered_file = osp.join(model_dir, f"taxonomy_filtered.json")

classe_ids = [x for x in list(os.listdir(model_dir)) if not osp.isfile(osp.join(model_dir, x)) ]

with open(taxonomy_file, 'r') as f:
    json_tax = json.load(f)

id2name_map = {}

for instance in json_tax:
    id2name_map[instance["synsetId"]] = instance["name"]

name2id_map = {}

for class_id in classe_ids:
    if class_id in id2name_map:
        name2id_map[id2name_map[class_id]] = class_id

with open(json_path_classes_shapenet, 'w') as f:
    json.dump(name2id_map, f, indent=4)


name2id_map_lack = {}
for class_id in id2name_map.keys():
    if class_id not in classe_ids:
        name2id_map[id2name_map[class_id]] = class_id

with open(json_path_classes_shapenet_lack, 'w') as f:
    json.dump(name2id_map_lack, f, indent=4)

