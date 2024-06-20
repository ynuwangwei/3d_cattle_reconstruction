import open3d as o3d
import numpy as np
import copy


pcd_top=o3d.io.read_point_cloud('./output_images/top.ply')
pcd_left=o3d.io.read_point_cloud('./output_images/left.ply')
pcd_right=o3d.io.read_point_cloud('./output_images/right.ply')

pcd_left.transform([
    [1.0, 0.0, 0.0, 0.49],
    [0.0, 0.12, 0.99, -2.45],
    [0.0, -0.99, 0.12, 1.84],
    [0.0, 0.0, 0.0, 1.0]
])

pcd_right.transform([
    [-1.0, -0.01, 0.09, 0.02],
    [-0.09, 0.0, -1.0, 2.22],
    [0.02, -1.0, 0.0, 2.17],
    [0.0, 0.0, 0.0, 1.0]
])

o3d.visualization.draw_geometries([pcd_top, pcd_left, pcd_right])