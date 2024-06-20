import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pcd_top=o3d.io.read_point_cloud('./output_images/top.ply')
pcd_left=o3d.io.read_point_cloud('./output_images/left.ply')

# 下采样
voxel_size=0.05
down_pcd_top=pcd_top.voxel_down_sample(voxel_size=voxel_size)
down_pcd_left=pcd_left.voxel_down_sample(voxel_size=voxel_size)

# 将top设为目标点云，计算法向量
down_pcd_top.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30
    )
)

# 聚类
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    left_labels=np.array(
        down_pcd_left.cluster_dbscan(eps=0.075,min_points=10,print_progress=True)
    )
    top_labels=np.array(
        down_pcd_top.cluster_dbscan(eps=0.1,min_points=10,print_progress=True)
    )

left_max_label=left_labels.max()
top_max_label=top_labels.max()
print(f'left side point cloud has {left_max_label+1} clusters')
print(f'top side point cloud has {top_max_label+1} clusters')

# 点云上色
left_colors=plt.get_cmap('tab20')(left_labels/(left_max_label if left_max_label>0 else 1))
left_colors[left_labels<0]=0
down_pcd_left.colors=o3d.utility.Vector3dVector(left_colors[:,:3])

top_colors=plt.get_cmap('tab20')(top_labels/(top_max_label if top_max_label>0 else 1))
top_colors[top_labels<0]=0
down_pcd_top.colors=o3d.utility.Vector3dVector(left_colors[:,:3])

o3d.visualization.draw_geometries([down_pcd_top],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215]
                                  )
