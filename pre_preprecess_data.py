import open3d as o3d
import numpy as np
import logging
import glob
import json
import os

def get_meshObj_and_scale(obj_file):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    xyz_max = [max([x[i] for x in mesh.vertices]) for i in range(3)]
    xyz_min = [min([x[i] for x in mesh.vertices]) for i in range(3)]
    xyz_cen = [-(xyz_max[i] + xyz_min[i]) / 2 for i in range(3)]
    xyz_scale = [(xyz_max[i] - xyz_min[i]) / 2 for i in range(3)]
    mesh.compute_vertex_normals()
    return mesh, xyz_cen, xyz_scale

def process(old_instance_path, new_instance_path):
    if not os.path.exists(new_instance_path):
        os.makedirs(new_instance_path)
    else:
        return
    old_mesh_filenames = list(glob.iglob(old_instance_path + "/**/*.obj")) + list(glob.iglob(old_instance_path + "/*.obj"))
    if len(old_mesh_filenames) != 1:
        return
    old_mesh_filename = old_mesh_filenames[0]
    mesh, xyz_cen, xyz_scale = get_meshObj_and_scale(old_mesh_filename)

    ds = [1 / x for x in xyz_scale]
    Rs = np.array([[ds[0],0,0],[0,ds[1],0],[0,0,ds[2]]])
    Rt = np.dot(Rs, xyz_cen)
    T_on = np.eye(4)
    T_on[:3,:3] = Rs
    T_on[:3,3] = Rt
    mesh.transform(T_on)
    T_no = np.linalg.inv(T_on)

    Trans = {}
    Trans['T_on'] = [list(x) for x in T_on]
    Trans['T_no'] = [list(x) for x in T_no]
    Trans['center'] = xyz_cen
    Trans['scale'] = xyz_scale
    Trans['scale_inv'] = ds

    b = json.dumps(Trans)


    o3d.io.write_triangle_mesh(os.path.join(new_instance_path, "model_normalized.obj"), mesh)
    f2 = open(os.path.join(new_instance_path, 'Transformation.json'), 'w')
    f2.write(b)
    f2.close()

if __name__ == "__main__":
    dataset_path = "/media/lj/TOSHIBA/dataset/ShapeNet"

    old_sname = "ShapeNetCore.v2"
    new_sname = "normed_ShapeNetCore.v2"
    source_name = "ShapeNetV2"

    old_source_dir = os.path.join(dataset_path, old_sname)
    new_source_dir = os.path.join(dataset_path, new_sname)
    # model_path = "04256520/1a4a8592046253ab5ff61a3a2a0e2484/models"

    split_file = "examples/splits/sv2_chairs_train.json"
    with open(split_file, "r") as f:
        json_data = json.load(f)
        class_directories = json_data[source_name]
        for class_dir in class_directories:
            old_class_path = os.path.join(old_source_dir, class_dir)
            new_class_path = os.path.join(new_source_dir, class_dir)
            # print(class_path)
            instance_dirs = class_directories[class_dir]
            logging.debug(
                "Processing " + str(len(instance_dirs)) + " instances of class " + class_dir
            )
            ins_num = len(instance_dirs)
            for i in range(ins_num):
                print("Propossing " , i, "/", ins_num)
                instance = instance_dirs[i]
                old_instance_path = os.path.join(old_class_path, instance)
                new_instance_path = os.path.join(new_class_path, instance)
                process(old_instance_path, new_instance_path)
