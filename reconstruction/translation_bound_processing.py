import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from solve_airbag import compute_ab_frames


DEBUG_MODE = False


def check_translation_bound(head_traj, ab_transx, ab_transy, special_interval,
                            dim_x_reproject=540, dim_y_reproject=960):
    """
    scale the translation to reach the bound
    """
    print(f"scale to match airbag pose = ({ab_transx}, {ab_transy})")
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    ab = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    pcd.translate(np.array([0, 0, 0]), relative=False)

    start_ab, _ = compute_ab_frames()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.get_view_control().set_zoom(1.5)
    ab.translate([0, -ab_transx, -ab_transy])
    head_x_pos = []
    head_y_pos = []
    original_pos = pcd.get_center()

    for counter in range(len(head_traj)):
        pcd.translate(head_traj[counter % len(head_traj)])
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        head_x_pos.append(pcd.get_center()[1])
        head_y_pos.append(pcd.get_center()[2])

    if special_interval is not None:
        print(" scaling specifically using an interval")
        start, end = map(int, special_interval)
        mi = np.min(head_x_pos[start:end])
        head_x_pos = [du - ab_transx-mi for du in head_x_pos]
        scale_y = abs((abs(ab_transy)) / np.min(head_y_pos))
        head_y_pos = [du * scale_y for du in head_y_pos]

    # recompute trajectory
    prev_pos = original_pos[1:]
    trajectories = []
    for idx in range(len(head_x_pos)):
        mean = np.array([head_x_pos[idx]*dim_x_reproject, head_y_pos[idx]*dim_y_reproject])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = move[1]
            trans[1] = move[0]
            trajectories.append(trans)
        prev_pos = mean

    # re-simulate
    new_head_x_pos = []
    new_head_y_pos = []
    pcd.translate(np.array([0, 0, 0]), relative=False)
    if DEBUG_MODE:
        for counter in range(len(trajectories)):
            pcd.translate(trajectories[counter % len(head_traj)])
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            new_head_x_pos.append(pcd.get_center()[1])
            new_head_y_pos.append(pcd.get_center()[2])
        plt.subplot(211)
        plt.plot(new_head_x_pos)
        if special_interval is not None:
            s, e = special_interval
            plt.plot([s, e-2], [new_head_x_pos[du] for du in [s, e-2]], "bo")
        plt.plot([-ab_transx*dim_x_reproject]*len(new_head_y_pos))

        plt.subplot(212)
        plt.plot(new_head_y_pos)
        if special_interval is not None:
            s, e = special_interval
            plt.plot([s, e-2], [new_head_y_pos[du] for du in [s, e-2]], "bo")
        plt.plot([-ab_transy*dim_y_reproject]*len(new_head_y_pos))

        plt.savefig("trans_bound.png")
        plt.close()

    vis.destroy_window()
    return trajectories


