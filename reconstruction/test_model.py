import open3d as o3d
import numpy as np
import skimage.io
import sys
import cv2

import meshio

mesh = meshio.read(
    "../data/model/model.obj",  # string, os.PathLike, or a buffer/open file
    # file_format="stl",  # optional if filename is a path; inferred from extension
    # see meshio-convert -h for all possible formats
)

text_coords = mesh.point_data["obj:vt"]
vert_coords = mesh.points
faces = mesh.cells_dict["triangle"]
print(text_coords.shape, vert_coords.shape, faces.shape)

pcd = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vert_coords), o3d.utility.Vector3iVector(faces))
txt_img = skimage.io.imread("../data/model/textures/Head_albedo.jpg")/255.0
pcd.compute_vertex_normals()
#
# pcd_vertices = np.asarray(pcd.vertices)
# colors = np.ones((pcd_vertices.shape[0], 3))*0.5
#
# uv_maps = np.asarray(pcd.triangle_uvs)
# pcd_tri = np.asarray(pcd.triangles)
#
# print(pcd_tri.shape)
# print(uv_maps.shape)
#
# print(uv_maps[:6])
#
#
def access_color(img, uv):
    u, v = uv*1024
    print(u, v)
    return img[1023-int(v), 1023-int(u)]
#
#
# for i in range(pcd_tri.shape[0]):
#     v1, v2, v3 = pcd_tri[i]
#     uv1 = uv_maps[i*3]
#     uv2 = uv_maps[i*3+1]
#     uv3 = uv_maps[i*3+2]
#     colors[v1] = access_color(txt_img, uv1)
#     colors[v2] = access_color(txt_img, uv2)
#     colors[v3] = access_color(txt_img, uv3)
#     print(v1, v2, v3, uv1, uv2, uv3)
#     if i > 2:
#         break

colors = np.zeros_like(text_coords)
for i in range(text_coords.shape[0]):
    colors[i] = access_color(txt_img, text_coords[i, :2])

pcd.vertex_colors = o3d.utility.Vector3dVector(colors)
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.get_view_control().set_zoom(1.5)
vis.run()
