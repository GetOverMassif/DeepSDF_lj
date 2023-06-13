import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import open3d as o3d
import scripts.show_npz as show_npz
import json

import argparse

# def getCoordinateAxis(T = np.eye(4), frameSize = 0.5):
#     gCoordinateAxis = o3d.geometry.TriangleMesh.create_coordinate_frame().\
#             scale(frameSize, np.array([0., 0., 0.]))
#     return gCoordinateAxis.transform(T)

# def get_xyz_boundary(pcd_surfacePts):
#     [x1,y1,z1,x2,y2,z2] = [-10000,-10000,-10000,10000,10000,10000]
#     for p in list(np.asarray(pcd_surfacePts.points)):
#         x1 = max(x1,p[0])
#         y1 = max(y1,p[1])
#         z1 = max(z1,p[2])
#         x2 = min(x2,p[0])
#         y2 = min(y2,p[1])
#         z2 = min(z2,p[2])
#     return [x1,y1,z1], [x2,y2,z2]

# def get_cube_lineset(xyz_max, xyz_min, color, xyz_centroid = [0,0,0]):
#     [x1, y1, z1] = xyz_max
#     [x2, y2, z2] = xyz_min
#     d = xyz_centroid
#     points = [[x1,y1,z2],[x1,y2,z2],[x2,y2,z2],[x2,y1,z2],[x1,y1,z1],[x1,y2,z1],[x2,y2,z1],[x2,y1,z1]]
#     points = [[(p[i] - d[i]) for i in range(3)] for p in points]
#     pointPairs = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]]
#     cube_lineset = o3d.geometry.LineSet()
#     cube_lineset.lines = o3d.utility.Vector2iVector(pointPairs)
#     cube_lineset.paint_uniform_color(color)
#     cube_lineset.points = o3d.utility.Vector3dVector(points)
#     return cube_lineset

# def get_meshObj_and_scale(obj_file, color):
#     mesh = o3d.io.read_triangle_mesh(obj_file)
#     xyz_max = [max([x[i] for x in mesh.vertices]) for i in range(3)]
#     xyz_min = [min([x[i] for x in mesh.vertices]) for i in range(3)]
#     mesh.compute_vertex_normals()
#     mesh.paint_uniform_color(color)
#     return mesh, xyz_max, xyz_min

# def get_meshObj(obj_file, color, T = np.eye(4)):
#     mesh = o3d.io.read_triangle_mesh(obj_file)
#     mesh.compute_vertex_normals()
#     mesh.paint_uniform_color(color)
#     mesh.transform(T)
#     # # obj顶点显示
#     # pcobj = o3d.geometry.PointCloud()
#     # pcobj.points = o3d.utility.Vector3dVector(mesh.vertices)
#     # o3d.visualization.draw_geometries([pcobj], window_name="Open3D2")
#     return mesh

# def T_scale(s, t = [0,0,0]):
#     T = np.eye(4)
#     for i in range(3):
#         T[i,i] = s
#         T[i,3] = t[i] * s
#     return T

# def save_view_point(vis, filename = "viewpoint.json"):
#     param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#     o3d.io.write_pinhole_camera_parameters(filename, param)

# def load_view_point(vis, filename = "viewpoint.json"):
#     ctr = vis.get_view_control()
#     param = o3d.io.read_pinhole_camera_parameters(filename)
#     ctr.convert_from_pinhole_camera_parameters(param)

def main(file_path):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name = 'Show SDF samples')

    # axis = getCoordinateAxis(np.eye(4), 0.6)
    # vis.add_geometry(axis)

    mgSDFSapsIn, mgSDFSapsOut, xyz_max, xyz_min = show_npz.get_npz_points(file_path)

    vis.add_geometry(mgSDFSapsIn, True)

    vis.run()
    vis.destroy_window()

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--sdf_file', type=str, required=True, help='sdf file path')
    args = argparser.parse_args()
    main(args.sdf_file)