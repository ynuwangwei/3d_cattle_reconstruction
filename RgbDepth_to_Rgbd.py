import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

color_raw=cv2.imread('./raw_images/left_rgb.png')
# 深度图的尺寸是(512,424),RGB图要与之对齐
color_raw=cv2.resize(color_raw, (512, 424))
color_raw=o3d.geometry.Image(color_raw)
# 读取深度图像
depth_raw=o3d.io.read_image('./raw_images/left_depth.png')

# 合成RGDB图像
rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw,
    depth_raw,
    convert_rgb_to_intensity=False
)

# 展示RGB图像和深度图像
plt.subplot(1,2,1)
plt.title("RGB image")
plt.imshow(color_raw)
plt.subplot(1,2,2)
plt.title("Depth image")
plt.imshow(depth_raw)
plt.show()

# RGBD图像生成点云
pcd=o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
    )
)

pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd])