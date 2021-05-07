import open3d as o3d
import numpy as np
import copy
import sys


sys.stdin = open("../data_heavy/frames/info.txt")
lines = [du[:-1] for du in sys.stdin.readlines()]
dense_corr_dir = "../data_heavy/matching_solutions"
images_dir = "../data_heavy/frames_ear_only_nonblack_bg"
saved_pc_dir = "../data_heavy/point_cloud_solutions"

pcd_list = []
vis = o3d.visualization.Visualizer()
vis.create_window()
for identifier in lines:
    pcd = o3d.io.read_point_cloud("%s/point_cloud-%s.txt" % (saved_pc_dir, identifier), "xyzrgb")
    pcd_list.append(pcd)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

o3d.visualization.draw_geometries(pcd_list)
# vis.poll_events()
# vis.update_renderer()
# vis.destroy_window()
