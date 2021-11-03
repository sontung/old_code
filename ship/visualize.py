import argparse
import cv2
import open3d as o3d
import kmeans1d
from os import listdir
from os.path import join
from test_model import new_model
from scipy.spatial.transform import Rotation as rot_mat_compute
from run_videos import read_json
from solve_position import compute_head_ab_areas


def visualize_wo_ab(comp1, comp2):
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    head_rot = comp1["head rot"]

    rotated_trajectory = [head_rot]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # start writing visualizations
    rot1 = rotated_trajectory[0]
    if rot1 > 90:
        rot_mat2 = rot_mat_compute.from_euler('x', -rot1 + 90,
                                              degrees=True).as_matrix()
    else:
        rot_mat2 = rot_mat_compute.from_euler('x', rot1 - 90,
                                              degrees=True).as_matrix()

    rot2 = comp2["head rot"]
    rot_mat = rot_mat_compute.from_euler('z', rot2-90,
                                         degrees=True).as_matrix()
    pcd.rotate(rot_mat, pcd.get_center())

    pcd.rotate(rot_mat2, pcd.get_center())
    vis.update_geometry(pcd)

    vis.run()
    vis.destroy_window()
    return cv2.imread("v1.png"), cv2.imread("v2.png")


def visualize(comp1, comp2):
    pcd = new_model()
    pcd.translate([0, 0, 0], relative=False)
    pcd.compute_vertex_normals()
    du_outputs, du_outputs2, (_, _) = compute_head_ab_areas(comp1)
    _, ab_transx, ab_transy, ab_rot = du_outputs2

    rotated_trajectory = du_outputs["rot trajectory"]
    global_head_scale = du_outputs["head scale"]
    global_ab_scale = du_outputs["ab scale"]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)

    # start writing visualizations
    pcd.scale(global_head_scale, pcd.get_center())
    rot1 = rotated_trajectory[0]
    if rot1 > 90:
        rot_mat2 = rot_mat_compute.from_euler('x', -rot1+90,
                                              degrees=True).as_matrix()
    else:
        rot_mat2 = rot_mat_compute.from_euler('x', rot1-90,
                                              degrees=True).as_matrix()
    pcd.rotate(rot_mat2, pcd.get_center())

    rot2 = comp2["head rot"]
    rot_mat = rot_mat_compute.from_euler('z', rot2-90,
                                         degrees=True).as_matrix()
    pcd.rotate(rot_mat, pcd.get_center())
    vis.update_geometry(pcd)

    ab = o3d.io.read_triangle_mesh(f"../new_particles_63.obj")
    ab.translate([0, 0, 0], relative=False)
    ab.compute_vertex_normals()
    ab.scale(global_ab_scale, ab.get_center())
    ab.translate([0, ab_transx, ab_transy])
    ab.rotate(rot_mat_compute.from_euler("y", 90, degrees=True).as_matrix())

    if ab_rot > 90:
        rot_mat2 = rot_mat_compute.from_euler('x', -ab_rot + 90,
                                              degrees=True).as_matrix()
    else:
        rot_mat2 = rot_mat_compute.from_euler('x', ab_rot - 90,
                                              degrees=True).as_matrix()
    ab.rotate(rot_mat2, ab.get_center())

    # view 2
    rot2 = comp2["ab rot"]
    rot_mat = rot_mat_compute.from_euler('z', rot2 - 90,
                                         degrees=True).as_matrix()
    ab.rotate(rot_mat, ab.get_center())

    trans = comp2["ab center"] - comp2["head center"]
    ab.translate([trans[0], trans[1], 0])

    vis.add_geometry(ab, reset_bounding_box=False)

    vis.run()
    vis.destroy_window()
    return cv2.imread("v1.png"), cv2.imread("v2.png")


def process(mypath='all_video'):
    """
    extract frames from videos stored in ../data_heavy/run
    """
    all_folders = [f for f in listdir(mypath)]

    # check noise
    pixels = []
    for count, folder in enumerate(all_folders):
        all_files = [join(join(mypath, folder), f) for f in listdir(join(mypath, folder))]
        for video in all_files:
            name = "%s-%s.png" % (folder, video.split('/')[-1])
            json_name = "%s-%s.json" % (folder, video.split('/')[-1])
            img = cv2.imread(f"images/{name}")
            computation = read_json(f"images/{json_name}", img)
            view = name.split(".mp4.png")[0].split("-")[-1]
            if len(computation) == 0:
                continue
            if view == "1":
                pixels.append(computation["head pixels"])
    u = kmeans1d.cluster(pixels, 2)
    head_pixels_lower_bound = u.centroids[0]

    for count, folder in enumerate(all_folders):
        if folder != FOLDER:
            continue
        print("processing", folder)
        all_files = [join(join(mypath, folder), f) for f in listdir(join(mypath, folder))]
        res = {}
        for video in all_files:
            name = "%s-%s.png" % (folder, video.split('/')[-1])
            json_name = "%s-%s.json" % (folder, video.split('/')[-1])
            img = cv2.imread(f"images/{name}")
            computation = read_json(f"images/{json_name}", img)
            view = name.split(".mp4.png")[0].split("-")[-1]
            if len(computation) == 0:
                continue
            if view == "1":
                if computation["head center"] is None:
                    computation["head pixels"] = computation["ab pixels"]
                    computation["head center"] = computation["ab center"]
                    computation["head rot"] = computation["ab rot"]
                if computation["head pixels"] <= head_pixels_lower_bound:
                    computation["head pixels"] = computation["ab pixels"]
                    computation["head center"] = computation["ab center"]

            res[view] = [computation, img]
        if len([u[0] for u in res.values()]) == 0:
            continue
        # check if airbag exists
        if sum([u[0]["ab pixels"] for u in res.values()]) == 0:
            airbag = False
        else:
            airbag = True

        if not airbag:
            _, _ = visualize_wo_ab(res["1"][0], res["2"][0])
        else:
            _, _ = visualize(res["1"][0], res["2"][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, default=False, help='input folder', required=True)
    args = vars(parser.parse_args())
    FOLDER = args['folder']
    if "/" in FOLDER:
        FOLDER = FOLDER.split("/")[-1]
    process()
