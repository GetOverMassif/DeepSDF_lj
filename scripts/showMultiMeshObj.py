import open3d as o3d
import numpy as np
import os

def save_view_point(vis, filename = "viewpoint.json"):
    cam_param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, cam_param)

def load_view_point(vis, filename = "viewpoint.json"):
    ctr = vis.get_view_control()
    cam_param = o3d.io.read_pinhole_camera_parameters(filename)
    ctr.convert_from_pinhole_camera_parameters(cam_param)

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

        self.model_types = ['raw', 'normed']
        self.model_type = 'raw'

        if self.model_type == 'raw':
            self.file_path = "/media/lj/TOSHIBA/dataset/ShapeNet/ShapeNetCore.v2/03001627"
        elif self.model_type =='normed':
            self.file_path = "/media/lj/TOSHIBA/dataset/ShapeNet/normed_ShapeNetCore.v2/03001627"
        else:
            raise TypeError

        dirs = os.listdir(self.file_path)
        self.instances = []

        # for dataname in dirs:
        #     print("dataname: ", dataname)
        #     if os.path.splitext(dataname)[1] == '.obj':
        #         self.instances.append(dataname)
            
        self.instances = list(dirs)
        self.instance_num = len(self.instances)
        self.current_idx = -1

        self.updateGeometry(1, True)
        
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

    def updateGeometry(self, step = 0, change_view=False):
        print("")
        
        if step == 0:
            self.current_idx = np.random.randint(1, self.instance_num)
        else:
            self.current_idx = (self.current_idx + int(step)) % self.instance_num
        self.cuModelIdx = os.path.splitext(self.instances[self.current_idx])[0]

        self.vis.clear_geometries()
        print("file:", self.current_idx,"/", self.instance_num,",",\
                ":", self.cuModelIdx)
        
        i_max, j_max = 4, 3
        interval_size = 1.3
        bbox_size = 0.4*interval_size

        for i in range(i_max):
            for j in range(j_max):
                x, y, z = i * interval_size, j * interval_size, 0
                pos = [x, y, z]
                index = (i * j_max + j + self.current_idx) % self.instance_num
                cuModelIdx = os.path.splitext(self.instances[index])[0]
                self.showMeshTargetPos(cuModelIdx, pos, change_view, bbox_size)

    def showMeshTargetPos(self, cuModelIdx, pos, change_view, bbox_size=1):
        x, y, z = pos

        if self.model_type == 'raw':
            file_name = os.path.join(self.file_path, cuModelIdx + "/models/model_normalized.obj")
        elif self.model_type =='normed':
            file_name = os.path.join(self.file_path, cuModelIdx + "/model_normalized.obj")
        
        mesh = o3d.io.read_triangle_mesh(file_name)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([i/255 for i in [112, 128, 144]])

        T = np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]])
        mesh.transform(T)
        axis = getCoordinateAxis(T, 1.5 * bbox_size)
        cube = get_cube_lineset([bbox_size+x,bbox_size+y,bbox_size+z], [-bbox_size+x,-bbox_size+y,-bbox_size+z], [0,0,0])
        self.vis.add_geometry(mesh, change_view)
        self.vis.add_geometry(axis, change_view)
        self.vis.add_geometry(cube, change_view)

if __name__ == "__main__":
    visualizer = Visualizer()


