import open3d as o3d
import numpy as np
import os, sys
sys.path.append("..")
from scripts.show_npz import get_npz_points

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
    showOutPts = True

    def __init__(self) -> None:
        self.vis.create_window(window_name='Show Mesh file')
        self.vis.register_key_callback(ord('D'), self.step_forward)  # A
        self.vis.register_key_callback(ord('A'), self.step_backward)  # D
        self.vis.register_key_callback(ord('X'), self.step_random)  # R
        self.vis.register_key_callback(ord('O'), self.switchOutPts)  # R
        self.vis.register_key_callback(ord('I'), self.stepInput)  # R
        # self.vis.register_key_callback(ord('R'), self.exitVis)  # R

        self.file_path = "/media/lj/TOSHIBA/dataset/ShapeNet/normal_data/SdfSamples/ShapeNetV2/03001627"
        # self.file_path = "/media/lj/TOSHIBA/dataset/ShapeNet/data/SdfSamples/ShapeNetV2/04256520"
        self.instances = os.listdir(self.file_path)
        self.instance_num = len(self.instances)
        self.current_idx = -1

        # with open("CheckedMesh.txt","r") as f:
        #     self.checked_meshes = f.read().split()
        
        T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        self.axis = getCoordinateAxis(T, 1.5)

        a1, a2 = 1, 0.6
        self.cube = get_cube_lineset([a1,a1,a1], [-a1,-a1,-a1], [0,0,0])
        self.cube2 = get_cube_lineset([a2,a2,a2], [-a2,-a2,-a2], [0,0,0])

        # self.updateGeometry(1)
        self.stepInput(self.vis)
        
        self.vis.run()
        self.vis.destroy_window()
    
    # def exitVis(self, vis):
    #     print("Writing into CheckedMesh.txt.")
    #     with open("CheckedMesh.txt","w") as f:
    #         for meshId in self.checked_meshes:
    #             f.write(str(meshId)+"\n")
    
    def step_forward(self, vis):
        save_view_point(self.vis)
        self.updateGeometryWithViewKept(1)
        load_view_point(self.vis)

    def step_backward(self, vis):
        save_view_point(self.vis)
        self.updateGeometryWithViewKept(-1)
        load_view_point(self.vis)
    
    def step_random(self, vis):
        save_view_point(self.vis)
        self.updateGeometryWithViewKept(0)
        load_view_point(self.vis)
    
    def stepInput(self, vis):
        save_view_point(self.vis)
        target_id = input("Target ID:")
        target_file = target_id + ".npz"
        if target_file in self.instances:
            step = self.instance_num + self.instances.index(target_file) - self.current_idx
            self.updateGeometry(step)
        else:
            print("No file : ", target_file)
        load_view_point(self.vis)

    def switchOutPts(self, vis):
        if self.showOutPts:
            self.vis.remove_geometry(self.gPtsOut, False)
        else:
            self.vis.add_geometry(self.gPtsOut, False)
        self.showOutPts = not self.showOutPts

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
        # self.cuModelIdx = os.path.splitext(self.instances[self.current_idx])[0]
        self.cuModelIdx = self.instances[self.current_idx]
        # if self.cuModelIdx not in self.checked_meshes:
        #     self.checked_meshes.append(self.cuModelIdx)
        # else:
        #     self.updateGeometry(1)
        self.vis.clear_geometries()
        print("file:", self.current_idx,"/", self.instance_num,",",\
                # len(self.checked_meshes), "checked",\
                ":", self.cuModelIdx,)
        file_name = os.path.join(self.file_path, self.cuModelIdx)
        self.gPtsIn, self.gPtsOut, _, _ = get_npz_points(file_name)
        # print(len(mesh.vertices))
        self.vis.add_geometry(self.gPtsIn)
        if self.showOutPts:
            self.vis.add_geometry(self.gPtsOut)
        self.vis.add_geometry(self.axis)
        self.vis.add_geometry(self.cube)
        self.vis.add_geometry(self.cube2)

if __name__ == "__main__":
    visualizer = Visualizer()


