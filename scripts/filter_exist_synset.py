
import json
import os.path as osp

json_path = f"/media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2"
raw_json_filename = osp.join(json_path, "taxonomy.json")
filt_json_filename = osp.join(json_path, "taxonomy_filtered.json")

f_raw = open(raw_json_filename, 'r')
raw_json = json.load(f_raw)
f_raw.close()

f_filt = open(filt_json_filename, 'w')

filtered_list = []

for syn_dict in raw_json:
    if osp.exists(osp.join(json_path, syn_dict['synsetId'])):
        filtered_list.append(syn_dict)

json.dump(filtered_list, f_filt, indent=2)
f_filt.close()

# print(type(raw_json))