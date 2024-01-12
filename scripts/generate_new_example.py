import os
import os.path as osp
import json
import subprocess
import argparse


if __name__=="__main__":
    shapenetcore2_path = f"/media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2"
    data_save_path = f"/media/lj/TOSHIBA/dataset/ShapeNet/data"

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--class_name", required=str, default="chairs", help="class to train")
    argparser.add_argument("-s", "--model_source", required=str, default=shapenetcore2_path, help="class to train")
    argparser.add_argument("-d", "--data_save_path", required=str, default=data_save_path, help="class to train")
    args = argparser.parse_args()

    ShapeNetClassFile = f"ShapeNetClass.json"
    f_class = open(ShapeNetClassFile, 'r')
    catogory_synset_dict = json.load(f_class)
    f_class.close()

    # target_catogory = 'bottles'
    target_catogory = args.class_name
    shapenetcore2_path = args.model_source
    data_save_path = args.data_save_path

    
    limit_num = 50

    synset_number = catogory_synset_dict[target_catogory]
    cato_path = osp.join(shapenetcore2_path, synset_number)
    model_ids = os.listdir(cato_path)

    if limit_num > 0 and limit_num < len(model_ids):
        model_ids = model_ids[:limit_num]

    train_num = int(0.75 * len(model_ids))

    print(f"train_num = {train_num}")

    train_model_ids = model_ids[:train_num]
    test_model_ids = model_ids[train_num:]

    train_filename = f"examples/splits/sv2_{target_catogory}_train.json"
    test_filename = f"examples/splits/sv2_{target_catogory}_test.json"

    train_data = {"ShapeNetV2":{synset_number:train_model_ids}}
    test_data = {"ShapeNetV2":{synset_number:test_model_ids}}

    f_train = open(train_filename, 'w')
    json.dump(train_data, f_train, indent=2)
    f_train.close()

    f_test = open(test_filename, 'w')
    json.dump(test_data, f_test, indent=2)
    f_test.close()

    chair_spec_filename = f'examples/chairs/specs.json'
    target_spec_filename = f'examples/{target_catogory}/specs.json'

    target_spec_path = f'examples/{target_catogory}'
    cmd = f"mkdir -p {target_spec_path}"
    p = subprocess.Popen(cmd, shell=True)
    return_code = p.wait()

    f_example = open(chair_spec_filename, 'r')
    json_data = json.load(f_example)
    f_example.close()

    json_data['Description'] = [
        f"This experiment learns a shape representation for {target_catogory} ",
        f"using data from ShapeNet version 2."
    ]
    json_data['DataSource'] = data_save_path
    json_data['TrainSplit'] = train_filename
    json_data['TestSplit'] = test_filename
    f_target = open(target_spec_filename, 'w')
    json.dump(json_data, f_target, indent=2)
    f_target.close()


