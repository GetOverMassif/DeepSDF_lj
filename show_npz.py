import numpy as np
import open3d as o3d
import os

def get_npz_points(data_path):
    data = np.load(data_path)
    data_pos, data_neg = data['pos'], data['neg']
    pointsIn, colorsIn = [], []
    pointsOut, colorsOut = [], []

    sdf_t_In = 0.015
    s_c_In = 1 / sdf_t_In

    sdf_t_Out = 0.2
    s_c_Out = 1 / sdf_t_Out

    [x1,y1,z1,x2,y2,z2] = [-10000,-10000,-10000,10000,10000,10000]

    length = len(data_neg)
    for i in range(length):
        p = list(data_neg[i])
        x1 = max(x1,p[0])
        y1 = max(y1,p[1])
        z1 = max(z1,p[2])
        x2 = min(x2,p[0])
        y2 = min(y2,p[1])
        z2 = min(z2,p[2])
        sdf = min(abs(p[3]), sdf_t_In)
        pointsIn.append(p[:3])
        colorsIn.append([1-s_c_In*sdf,s_c_In*sdf,0])
        # colorsIn.append([1,s_c*sdf,0])
        print("    Inside points: ", i,"/",length, end = '\r')
    print("")
    length = len(data_pos)
    for i in range(length):
        p = list(data_pos[i])
        sdf = min(abs(p[3]), sdf_t_Out)
        pointsOut.append(p[:3])
        colorsOut.append([1-s_c_Out*sdf,s_c_Out*sdf,0])
        print("    Outside points: ", i,"/",length, end = '\r')
        # print([1-s_c*sdf,s_c*sdf,0])
    print("")

    gPtsIn = o3d.geometry.PointCloud()
    gPtsIn.points = o3d.utility.Vector3dVector(np.array(pointsIn)) # pos[:3]
    gPtsIn.colors = o3d.utility.Vector3dVector(colorsIn)

    gPtsOut = o3d.geometry.PointCloud()
    gPtsOut.points = o3d.utility.Vector3dVector(np.array(pointsOut)) # pos[:3]
    gPtsOut.colors = o3d.utility.Vector3dVector(colorsOut)

    x = min(abs(x1),abs(x2))
    y = min(abs(y1),abs(y2))
    z = min(abs(z1),abs(z2))
    [x1,y1,z1] = [x, y, z]
    [x2,y2,z2] = [-x, -y, -z]

    return gPtsIn, gPtsOut, [x1,y1,z1], [x2,y2,z2]

    # vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis.create_window(window_name='Show .npz file')
    # vis.add_geometry(pts_pcd)

    # vis.run()
    # vis.destroy_window()