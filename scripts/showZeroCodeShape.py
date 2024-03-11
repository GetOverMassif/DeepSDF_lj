import sys,os

sys.path.append("./")
# sys.path.append("~/Documents/codes/QSP-SLAM")


# import reconstruct
# import reconstruct.utils as io_utils
import reconstruct.optimizer as optim
import numpy as np
from reconstruct.utils import ForceKeyErrorDict
import open3d as o3d

# sys.path.append('./')


import matplotlib.pyplot as plt
import cv2
from deep_sdf_dsp.deep_sdf.workspace import config_decoder
# import dsc_slam_py.GetO3dGeometry as gog


def getO3dLineSet(points, pointPairs, color = [0,0,0], T = np.eye(4)):
    O3dLineSet = o3d.geometry.LineSet()
    O3dLineSet.lines = o3d.utility.Vector2iVector(pointPairs)
    O3dLineSet.points = o3d.utility.Vector3dVector(points)
    O3dLineSet.paint_uniform_color(color)
    return O3dLineSet.transform(T)

if __name__ == "__main__":
    
    # pyCfgPath = "configs/config_redwood_01053.json"
    pyCfgPath = "configs/config_redwood_01053_normed_chair.json"
    # # SequencePath = "/media/lj/TOSHIBA/dataset/DSP-SLAM/data/redwood_chairs/01053"
    # pyCfg = io_utils.get_configs(pyCfgPath)

    # pyDecoder = io_utils.get_decoder(pyCfg)
    # # pySequence = reconstruct2.get_sequence(SequencePath,pyCfg)
    # pyOptimizer = optim.Optimizer(pyDecoder,pyCfg)
    # code_len = pyCfg["optimizer"]["code_len"]
    # # code_len = 67
    # voxels_dim = pyCfg["voxels_dim"]

    decoder_path = f"/home/lj/Documents/codes/QSP-SLAM/weights/deepsdf/keyboards_64"

    pyDecoder = config_decoder(decoder_path)
    code_len = 64
    voxels_dim = 128
    pyMeshExtractor = optim.MeshExtractor(pyDecoder, code_len, voxels_dim)

    # # 双目模式，获取新观测 + 创建物体
    # # LocalMapping::GetNewObservations
    # SE3Tco = pyOptimizer.estimate_pose_cam_obj(SE3Tco_det, objScale_pMO, surfacePoints_det, objShapeCode_pMO)
    # # LocalMapping::CreateNewMapObjects
    # pyMapObject = pyOptimizer.reconstruct_object(Sim3Tco_det, surfacePoints_det, rayDirections_det, depthObs_det)
    # Sim3Tco = pyMapObject.t_cam_obj
    # code = pyMapObject.code

    # """show Detection Result"""
    # K = np.array([[541.425,    0.0, 320.0], \
    #               [    0.0,539.450, 240.0], \
    #               [    0.0,    0.0,   1.0]])
    # mnFrameId = 530  # KeyFrameId: 63
    # detections = pySequence.get_frame_by_id(mnFrameId)
    # print(type(detections))
    # print(type(detections[0]))
    # for det in detections:
    #     print("bbox = ", det.bbox)
    #     # print("type(bgd_rays) = ", type(det.background_rays))
    #     # print("len(bgd_rays) = ", len(det.background_rays))
    #     plt.title('mask')
    #     plt.imshow(det.mask)

    #     Pts = det.background_rays
    #     pts = np.dot(K, Pts.transpose())
    #     px, py = list(pts[0]), list(pts[1])
    #     # print(Pts)
    #     # pt = [x/P[2] for x in np.dot(self.K,P)][:2]
    #     # [pt_x, pt_y] = [int(x + 0.5) for x in pt]

    #     plt.plot(px,py,'o')
    #     plt.show()

    # # 单目模式
    # # LocalMapping::ProcessDetectedObjects
    print("line 70")
    SE3Tcw = np.eye(4)
    # SE3Tcw = np.array([[ 0.3115783, -0.2614453,  0.9135454,-2.1],
    #                    [ 0.8356056,  0.5331305, -0.1324205, 0.3],
    #                    [-0.4524182,  0.8046230,  0.3845771, 0.4],
    #                    [0.0000000,  0.0000000,  0.0000000, 1.0]])
    SE3Twc = np.linalg.inv(SE3Tcw)
    # Sim3Two_pMO = np.eye(4)
    # surface_points_cam = [[],[],[]]
    code1 = np.zeros(code_len, dtype='float32')
    # code1[64:] = [0.4,0.6,0.5]
    pyMapObject = ForceKeyErrorDict(t_cam_obj = np.eye(4), code = code1, is_good = True, loss = 0)
    # # # pyMapObject = pyOptimizer.reconstruct_object(SE3Tcw * Sim3Two_pMO, surface_points_cam, rays, depth_obs, vShapeCode_pMO)
    # # pyMapObject = pyOptimizer.reconstruct_object_by_surface_pts_only(SE3Tcw * Sim3Two_pMO, surface_points_cam, pyMapObject.code)
    # # for i in range(1,4):
    # #     flipped_Two = Sim3Two_pMO
    # #     flipped_Two[:3,0] = [-x for x in flipped_Two[:3,0]]
    # #     flipped_Two[:3,2] = [-x for x in flipped_Two[:3,2]]
    # #     # pyMapObjectFlipped = pyOptimizer.reconstruct_object(SE3Tcw * flipped_Two, surface_points_cam, rays, depth_obs, vShapeCode_pMO)
    # #     pyMapObject = pyOptimizer.reconstruct_object_by_surface_pts_only(SE3Tcw * Sim3Two_pMO, surface_points_cam, pyMapObject.code)
    Sim3Tco = pyMapObject.t_cam_obj
    # Sim3Tco = np.array([[0.2198463, -0.6040228,  0.7660444, 1.2],
    #                     [0.9447990, -0.0637250, -0.3213938,-0.8],
    #                     [0.2429454,  0.7944152,  0.5566704, 0.9],
    #                     [0.0000000,  0.0000000,  0.0000000, 1.0]])
    # # Sim3Tco[0,0:3] = [x * 0.5 for x in Sim3Tco[0,0:3]]
    # # Sim3Tco[1,0:3] = [x * 1.2 for x in Sim3Tco[1,0:3]]
    # # Sim3Tco[2,0:3] = [x * 1.7 for x in Sim3Tco[2,0:3]]

    code = pyMapObject.code
    Sim3Two = np.dot(SE3Twc, Sim3Tco)

    # [w,h,l] = [0.8, 1.1, 0.9]
    # # [w,h,l] = [0.9 for i in range(3)]
    # print("Sim3Two:\n",Sim3Two)
    # Sim3Two[0,0:3] = [x * w for x in Sim3Two[0,0:3]]
    # Sim3Two[1,0:3] = [x * h for x in Sim3Two[1,0:3]]
    # Sim3Two[2,0:3] = [x * l for x in Sim3Two[2,0:3]]

    # code = np.random.normal(0, 0.2, (code_len)).astype(np.float32)
    # code = np.array([-0.0138306, 0.01796659, -0.01468967, 0.0138555, 0.01339987, -0.00947178, 0.00326438, 0.01277796, \
    #                  -0.00117066,0.01341495, -0.0142548, -0.00351768,0.00136577,  0.01513615,-0.00758925,-0.00386562, \
    #                  -0.02131228,0.00072746, -0.01117157, 0.00116639, 0.0093996, -0.00433471,-0.00554136,-0.01497497, \
    #                   0.00900672,-0.00145061,-0.00822386, 0.00273526,-0.01267378, 0.00948004, 0.00969164,-0.00925022, \
    #                   0.00429466, 0.014799,  -0.01908216,-0.00369913, 0.00020886,-0.00884676, 0.00871749, 0.00772839, \
    #                   0.00073875, 0.00903708, 0.00167913, 0.01208792,-0.00476207,-0.00434479, 0.00678251, 0.00050789, \
    #                   0.00296459,-0.00479323, 0.02222087, 0.00466129, 0.00966943,-0.01266049,-0.00464642, 0.00540745, \
    #                  -0.00051983, 0.00650448, 0.00169991,-0.01669517,-0.01186821,-0.00728646, 0.00244802, 0.00252918]).astype(np.float32)

    print("code1 = \n", code1)

    pyMesh = pyMeshExtractor.extract_mesh_from_code(code1)
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(pyMesh.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(pyMesh.faces))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5,0.5,0])
    mesh.transform(Sim3Two)
    coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame().\
            scale(1.0, np.array([0., 0., 0.]))

    a1 = 0.6
    # x,y,z = 1, 1, 1
    # cubeLineSet1 = gog.getCubeLineset([a1,a1,a1], [-a1,-a1,-a1], [1,0,0])
    # cubeLineSet2 = gog.getCubeLineset([x,y,z], [-x,-y,-z], [0,0,0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name = 'Show object after optimization')
    vis.add_geometry(mesh)
    # vis.add_geometry(coor_frame)
    
    # vis.add_geometry(cubeLineSet1)
    # vis.add_geometry(cubeLineSet2)
    vis.run()
    vis.destroy_window()