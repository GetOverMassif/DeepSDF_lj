import open3d as o3d
import numpy as np
import os
from show_npz import get_npz_points

def save_view_point(vis, filename = "viewpoint.json"):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)

def load_view_point(vis, filename = "viewpoint.json"):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(param)

def getCoordinateAxis(T = np.eye(4), frameSize = 0.5):
    gCoordinateAxis = o3d.geometry.TriangleMesh.create_coordinate_frame().\
            scale(frameSize, np.array([0., 0., 0.]))
    return gCoordinateAxis.transform(T)

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

class Visualizer:
    vis = o3d.visualization.VisualizerWithKeyCallback()

    def __init__(self) -> None:
        self.vis.create_window(window_name='Show Mesh file')
        self.vis.register_key_callback(ord('D'), self.step_forward)  # A
        self.vis.register_key_callback(ord('A'), self.step_backward)  # D
        self.vis.register_key_callback(ord('X'), self.step_random)  # R
        # self.vis.register_key_callback(ord('R'), self.exitVis)  # R

        self.file_path = "/media/lj/TOSHIBA/dataset/ShapeNet/deformed_data/SdfSamples/ShapeNetV2/03001627/1007e20d5e811b308351982a6e40cf41"
        dirs = os.listdir(self.file_path)
        self.instances = []
        for dataname in dirs:
            print("dataname: ", dataname)
            if os.path.splitext(dataname)[1] == '.npz':
                self.instances.append(dataname)
        self.instance_num = len(self.instances)
        self.current_idx = -1

        # with open("CheckedMesh.txt","r") as f:
        #     self.checked_meshes = f.read().split()
        
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.axis = getCoordinateAxis(T, 1.5)
        self.cube = get_cube_lineset([1,1,1], [-1,-1,-1], [0,0,0])

        self.updateGeometry(1)
        
        self.vis.run()
        self.vis.destroy_window()
    
    # def exitVis(self, vis):
    #     print("Writing into CheckedMesh.txt.")
    #     with open("CheckedMesh.txt","w") as f:
    #         for meshId in self.checked_meshes:
    #             f.write(str(meshId)+"\n")
    
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
        
        if step == 0:
            self.current_idx = np.random.randint(1, self.instance_num)
        else:
            self.current_idx = (self.current_idx + int(step)) % self.instance_num
        self.cuModelIdx = os.path.splitext(self.instances[self.current_idx])[0]
        # if self.cuModelIdx not in self.checked_meshes:
        #     self.checked_meshes.append(self.cuModelIdx)
        # else:
        #     self.updateGeometry(1)
        self.vis.clear_geometries()
        print("file:", self.current_idx,"/", self.instance_num,",",\
                # len(self.checked_meshes), "checked",\
                ":", self.cuModelIdx,)
        # file_name = os.path.join(self.file_path, self.cuModelIdx + "/model_normalized.obj")
        file_name = os.path.join(self.file_path, str(self.current_idx) + ".npz")
        # file_name = os.path.join(self.file_path, self.cuModelIdx + "/models/model_normalized.obj")

        gPtsIn, gPtsOut, [x1,y1,z1], [x2,y2,z2] = get_npz_points(file_name)
        self.vis.add_geometry(gPtsIn, False)
        self.vis.add_geometry(gPtsOut, False)
        # print(len(mesh.vertices))

        # mesh.compute_vertex_normals()
        # mesh.paint_uniform_color([i/255 for i in [112, 128, 144]])
        
        self.vis.add_geometry(self.axis)
        self.vis.add_geometry(self.cube)

if __name__ == "__main__":
    visualizer = Visualizer()


