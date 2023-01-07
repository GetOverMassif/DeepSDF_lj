import numpy as np
import open3d as o3d
import os
import show_npz
import json

def getCoordinateAxis(T = np.eye(4), frameSize = 0.5):
    gCoordinateAxis = o3d.geometry.TriangleMesh.create_coordinate_frame().\
            scale(frameSize, np.array([0., 0., 0.]))
    return gCoordinateAxis.transform(T)

def get_xyz_boundary(pcd_surfacePts):
    [x1,y1,z1,x2,y2,z2] = [-10000,-10000,-10000,10000,10000,10000]
    for p in list(np.asarray(pcd_surfacePts.points)):
        x1 = max(x1,p[0])
        y1 = max(y1,p[1])
        z1 = max(z1,p[2])
        x2 = min(x2,p[0])
        y2 = min(y2,p[1])
        z2 = min(z2,p[2])
    return [x1,y1,z1], [x2,y2,z2]

def get_cube_lineset(xyz_max, xyz_min, color, xyz_centroid = [0,0,0]):
    [x1, y1, z1] = xyz_max
    [x2, y2, z2] = xyz_min
    d = xyz_centroid
    points = [[x1,y1,z2],[x1,y2,z2],[x2,y2,z2],[x2,y1,z2],[x1,y1,z1],[x1,y2,z1],[x2,y2,z1],[x2,y1,z1]]
    points = [[(p[i] - d[i]) for i in range(3)] for p in points]
    pointPairs = [[0,1],[1,2],[2,3],[3,0],[0,4],[1,5],[2,6],[3,7],[4,5],[5,6],[6,7],[7,4]]
    cube_lineset = o3d.geometry.LineSet()
    cube_lineset.lines = o3d.utility.Vector2iVector(pointPairs)
    cube_lineset.paint_uniform_color(color)
    cube_lineset.points = o3d.utility.Vector3dVector(points)
    return cube_lineset

def get_meshObj_and_scale(obj_file, color):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    xyz_max = [max([x[i] for x in mesh.vertices]) for i in range(3)]
    xyz_min = [min([x[i] for x in mesh.vertices]) for i in range(3)]
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh, xyz_max, xyz_min

def get_meshObj(obj_file, color, T = np.eye(4)):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.transform(T)
    # # obj顶点显示
    # pcobj = o3d.geometry.PointCloud()
    # pcobj.points = o3d.utility.Vector3dVector(mesh.vertices)
    # o3d.visualization.draw_geometries([pcobj], window_name="Open3D2")
    return mesh

def T_scale(s, t = [0,0,0]):
    T = np.eye(4)
    for i in range(3):
        T[i,i] = s
        T[i,3] = t[i] * s
    return T

def save_view_point(vis, filename = "viewpoint.json"):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)

def load_view_point(vis, filename = "viewpoint.json"):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)

class Visualizer:
    vis = o3d.visualization.VisualizerWithKeyCallback()
    shapeNet_path = "/media/lj/TOSHIBA/dataset/ShapeNet"

    recMesh_file_path = "examples/sofas/Reconstructions/10/Meshes/ShapeNetV2/04256520"
    recMesh_file_List = []
    recMesh_file_Num = 1
    cuRecMeshIdx = 0
    cuModelIdx = ""

    mgSDFSapsIn = o3d.geometry.PointCloud()
    mgSDFSapsOut = o3d.geometry.PointCloud()
    mgSurfacePts = o3d.geometry.PointCloud()
    mgOriMesh = o3d.geometry.TriangleMesh()
    mgRecMesh = o3d.geometry.TriangleMesh()

    startId = 37
    show_cube = False
    mode = [True, True, True, True, False]

    def __init__(self) -> None:
        self.vis.create_window(window_name = 'Show object in different formats')
        self.vis.register_key_callback(ord('1'), self.switchOriMeshState)  # O
        self.vis.register_key_callback(ord('2'), self.switchSDFSapsInState)  # S
        self.vis.register_key_callback(ord('3'), self.switchSurfPtsState)  # P
        self.vis.register_key_callback(ord('4'), self.switchRecMeshState)  # R
        self.vis.register_key_callback(ord('5'), self.switchSDFSapsOutState)  # 
        self.vis.register_key_callback(ord('A'), self.step_forward)  # R
        self.vis.register_key_callback(ord('D'), self.step_backward)  # R
        self.vis.register_key_callback(ord('X'), self.step_random)  # R

        self.recMesh_file_List = os.listdir(self.recMesh_file_path)
        self.recMesh_file_Num = len(self.recMesh_file_List)
        
        self.updateGeometry(self.startId)

        self.vis.run()
        self.vis.destroy_window()
    
    def step_forward(self, vis):
        self.updateGeometryWithViewKept(1)

    def step_backward(self, vis):
        self.updateGeometryWithViewKept(-1)
    
    def step_random(self, vis):
        self.updateGeometryWithViewKept(0)

    def updateGeometryWithViewKept(self, step):
        save_view_point(self.vis)
        self.updateGeometry(step)
        load_view_point(self.vis)

    def updateGeometry(self, step = 0):
        print("")
        self.vis.clear_geometries()
        if step == 0:
            self.cuRecMeshIdx = np.random.randint(1, self.recMesh_file_Num)
        else:
            self.cuRecMeshIdx = (self.cuRecMeshIdx + int(step)) % self.recMesh_file_Num
        self.cuModelIdx = os.path.splitext(self.recMesh_file_List[self.cuRecMeshIdx])[0]
        print("file:", self.cuRecMeshIdx,"/", self.recMesh_file_Num," : ", self.cuModelIdx)

        axis = getCoordinateAxis(np.eye(4), 0.6)
        self.vis.add_geometry(axis)
        norm_file_path = os.path.join(self.shapeNet_path, "data/NormalizationParameters/ShapeNetV2/04256520")
        norm_file = os.path.join(norm_file_path, self.cuModelIdx + ".npz")
        norm_data = np.load(norm_file)
        offset, scale = list(norm_data['offset']), float(norm_data['scale'])
        # T_st = T_scale(scale, offset)

        # 1. show .obj: origin object meshes
        print("1.[Origin object meshes] - O")
        oriMesh_file_path = os.path.join(self.shapeNet_path, "ShapeNetCore.v2/04256520")
        oriMesh_file = os.path.join(oriMesh_file_path, self.cuModelIdx + "/models/model_normalized.obj")
        json_path = os.path.join(oriMesh_file_path, self.cuModelIdx + "/models/model_normalized.json")

        # with open(json_path, "r") as f:
        #     json_data = json.load(f)
        #     xyz_max, xyz_min = json_data["max"], json_data["min"]
        #     s = [(xyz_max[i] - xyz_min[i]) for i in range(3)]  # s = [w,h,l]
        #     # whl_min = min(s)
        #     # ds = [whl_min / x for x in s]
        #     ds = [2 / x for x in s]
        #     T_st = np.array([[ds[0],0,0,0],[0,ds[1],0,0],[0,0,ds[2],0],[0,0,0,1]])
        #     # xyz_min_adj = [-x for x in xyz_max_adj]

        self.mgOriMesh, xyz_max, xyz_min = get_meshObj_and_scale(oriMesh_file, [i/255 for i in [112, 128, 144]])
        xyz_cen = [-(xyz_max[i] + xyz_min[i]) / 2 for i in range(3)]
        xyz_scale = [(xyz_max[i] - xyz_min[i]) / 2 for i in range(3)]

        ds = [1 / x for x in xyz_scale]
        Rs = np.array([[ds[0],0,0],[0,ds[1],0],[0,0,ds[2]]])
        Rt = np.dot(Rs, xyz_cen)
        T_st = np.eye(4)
        T_st[:3,:3] = Rs
        T_st[:3,3] = Rt
        self.mgOriMesh.transform(T_st)
        self.mgOriMeshCube = get_cube_lineset([1,1,1], [-1,-1,-1], [1,0,0])

        if self.mode[0]:
            self.vis.add_geometry(self.mgOriMesh)
            self.vis.add_geometry(self.mgOriMeshCube)
        
        # 2. show .npz: SDF samples
        print("2.[SDF Samples] - S")
        sdfSamples_file_path = os.path.join(self.shapeNet_path, "data/SdfSamples/ShapeNetV2/04256520/")
        sdfSamples_file = os.path.join(sdfSamples_file_path, self.cuModelIdx + ".npz")
        self.mgSDFSapsIn, self.mgSDFSapsOut, xyz_max, xyz_min = show_npz.get_npz_points(sdfSamples_file)
        cube_lineset = get_cube_lineset(xyz_max, xyz_min, [1.0,0.0,0.0])
        print("    cubeBox:", [float('{:.4f}'.format(i)) for i in xyz_max], \
                              [float('{:.4f}'.format(i)) for i in xyz_min])
        if self.mode[1]:
            self.vis.add_geometry(self.mgSDFSapsIn, False)
        if self.mode[4]:
            self.vis.add_geometry(self.mgSDFSapsOut, False)
        if self.show_cube:
            self.vis.add_geometry(cube_lineset, False)
        
        # 3. show .ply: surface points
        print("3.[Surface Points] - P")
        surfacePts_file_path = os.path.join(self.shapeNet_path, "data/SurfaceSamples/ShapeNetV2/04256520")
        surfacePts_file = os.path.join(surfacePts_file_path, self.cuModelIdx + ".ply")
        self.mgSurfacePts = o3d.io.read_point_cloud(surfacePts_file)
        xyz_max, xyz_min = get_xyz_boundary(self.mgSurfacePts)
        self.mgSurfacePts.paint_uniform_color([0.0,0.0,1.0])
        # offset, scale = [-(xyz_max[i] + xyz_min[i]) / 2 for i in range(3)], 2
        print("    offset:", [float('{:.4f}'.format(i)) for i in offset], \
                ", scale: ", float('{:.4f}'.format(scale)))
        xyz_max = [scale * (xyz_max[i] + offset[i]) for i in range(3)]
        xyz_min = [scale * (xyz_min[i] + offset[i]) for i in range(3)]
        self.mgSurfacePts.transform(T_st)
        cube_lineset = get_cube_lineset(xyz_max, xyz_min, [0.0,1.0,0.0])
        print("    cubeBox:", [float('{:.4f}'.format(i)) for i in xyz_max], \
                              [float('{:.4f}'.format(i)) for i in xyz_min])
        if self.mode[2]:
            self.vis.add_geometry(self.mgSurfacePts, False)
        if self.show_cube:
            self.vis.add_geometry(cube_lineset, False)
        
        # show .ply: reconstructed meshes
        print("4.[Reconstructed Object Meshes] - R")
        recMesh_file_path = "examples/sofas/Reconstructions/10/Meshes/ShapeNetV2/04256520"
        recMesh_file = os.path.join(recMesh_file_path, self.cuModelIdx + ".ply")
        self.mgRecMesh = get_meshObj(recMesh_file, [i/255 for i in [205, 170, 125]])
        print("    cubeBox:", [float('{:.4f}'.format(i)) for i in xyz_max], \
                              [float('{:.4f}'.format(i)) for i in xyz_min])
        if self.mode[3]:
            self.vis.add_geometry(self.mgRecMesh, False)
        if self.show_cube:
            self.vis.add_geometry(cube_lineset)

    def switchOriMeshState(self, vis):
        self.mode[0] = self.switchState(self.mode[0], self.mgOriMesh)
    
    def switchSDFSapsInState(self, vis):
        self.mode[1] = self.switchState(self.mode[1], self.mgSDFSapsIn)

    def switchSurfPtsState(self, vis):
        self.mode[2] = self.switchState(self.mode[2], self.mgSurfacePts)

    def switchRecMeshState(self, vis):
        self.mode[3] = self.switchState(self.mode[3], self.mgRecMesh)

    def switchSDFSapsOutState(self, vis):
        self.mode[4] = self.switchState(self.mode[4], self.mgSDFSapsOut)

    def switchState(self, state, mGeometry):
        if state:
            self.vis.remove_geometry(mGeometry, False)
        else:
            self.vis.add_geometry(mGeometry, False)
        return not state


if __name__ == "__main__":
    visualizer = Visualizer()
    