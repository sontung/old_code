import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from solve_airbag import compute_ab_frames


DEBUG_MODE = False


def check_translation_bound(head_traj, ab_transx, ab_transy):
    """
    scale the translation to reach the bound
    """
    print(f"scale to match airbag pose = ({ab_transx}, {ab_transy})")
    os.makedirs("../data_heavy/area_compute/", exist_ok=True)
    pcd = o3d.io.read_triangle_mesh("../data/max-planck.obj")
    ab = o3d.io.read_triangle_mesh("../data/max-planck.obj")

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

    if DEBUG_MODE:
        print("before", np.min(head_y_pos), np.max(head_y_pos), -ab_transy)

    scale_x = abs((abs(ab_transx))/np.min(head_x_pos))
    scale_y = abs((abs(ab_transy))/np.min(head_y_pos))

    head_x_pos_old = head_x_pos[:]
    head_y_pos_old = head_y_pos[:]

    head_x_pos = [du*scale_x for du in head_x_pos]
    head_y_pos = [du*scale_y for du in head_y_pos]

    # recompute trajectory
    prev_pos = original_pos[1:]
    trajectories = []
    for idx in range(len(head_x_pos)):
        mean = np.array([head_x_pos[idx], head_y_pos[idx]])
        if prev_pos is not None:
            trans = np.zeros((3, 1))
            move = mean - prev_pos
            trans[2] = move[1]
            trans[1] = move[0]
            trajectories.append(trans)
        prev_pos = mean

    # re-simulate
    if DEBUG_MODE:
        new_head_x_pos = []
        new_head_y_pos = []
        pcd.translate(original_pos-pcd.get_center())
        if DEBUG_MODE:
            print("at", pcd.get_center(), original_pos)
        for counter in range(len(trajectories)):
            pcd.translate(trajectories[counter % len(head_traj)])
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            new_head_x_pos.append(pcd.get_center()[1])
            new_head_y_pos.append(pcd.get_center()[2])
        print("after", np.min(new_head_y_pos), np.max(new_head_y_pos), -ab_transy)

        plt.subplot(211)
        plt.plot(head_x_pos_old)
        plt.plot(new_head_x_pos)
        plt.plot([-ab_transx]*len(new_head_x_pos))
        plt.legend(["ori", "scaled", "bound"])

        plt.subplot(212)
        plt.plot(head_y_pos_old)
        plt.plot(new_head_y_pos)
        plt.plot([-ab_transy]*len(new_head_y_pos))
        plt.legend(["ori", "scaled", "bound"])
        plt.savefig("trans_bound.png")
        plt.close()

    vis.destroy_window()
    return trajectories


