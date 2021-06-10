import sys
import os.path
from tqdm import tqdm

from rec_utils import b_spline_smooth
from scipy import interpolate
import numpy as np
import json
import glob


def sample_accordingly(new_time_step=27):
    all_json = glob.glob("../data_heavy/sph_solutions/state/*.json")
    all_json = sorted(all_json, key=lambda du: float(du.split("/")[-1].split("state_")[1].split("_particle")[0]))
    nb_time_step = len(all_json)
    max_nb_particles = -1
    all_particles_pos = {}
    for i, file1 in enumerate(all_json):
        with open(file1, "r") as fp:
            data = json.load(fp)
        particle_data = data["particles"]

        nb_particles = len(particle_data)
        if max_nb_particles < nb_particles:
            max_nb_particles = nb_particles

        for a_key in particle_data:
            particle = particle_data[a_key]
            particle_id = particle["id"][0]
            if particle_id not in all_particles_pos:
                all_particles_pos[particle_id] = np.full((nb_time_step, 3), -100, dtype=float)
            all_particles_pos[particle_id][i] = particle["position"]

    particles_matrix = np.full((max_nb_particles, nb_time_step, 3), -100, dtype=float)
    for i in range(max_nb_particles):
        pos = all_particles_pos[i]
        particles_matrix[i][:len(pos)] = pos

    new_particles_matrix = np.full((max_nb_particles, new_time_step, 3), -100, dtype=float)
    # sampling frame
    for i in tqdm(range(max_nb_particles), desc="Sampling new vertices"):
        p = particles_matrix[i]
        null_idx = np.argwhere(p[:, 0] <= -99)[:, 0]
        start_idx = 0
        if len(null_idx) != 0:
            start_idx = np.max(null_idx)+1
        tck_x = b_spline_smooth(p[start_idx:, 0], return_params=True)
        tck_y = b_spline_smooth(p[start_idx:, 1], return_params=True)
        tck_z = b_spline_smooth(p[start_idx:, 2], return_params=True)
        last_time_step = p[start_idx+1:, 2].shape[0]-1

        x = [interpolate.splev(du, tck_x) for du in np.linspace(0, last_time_step, new_time_step-start_idx, endpoint=True)]
        y = [interpolate.splev(du, tck_y) for du in np.linspace(0, last_time_step, new_time_step-start_idx, endpoint=True)]
        z = [interpolate.splev(du, tck_z) for du in np.linspace(0, last_time_step, new_time_step-start_idx, endpoint=True)]
        new_particles_matrix[i, start_idx:, 0] = x
        new_particles_matrix[i, start_idx:, 1] = y
        new_particles_matrix[i, start_idx:, 2] = z

        # import matplotlib.pyplot as plt
        # print(x)
        # print(particles_matrix[i, :, 0])
        # print(len(x), particles_matrix[i, :, 0].shape)
        # print(np.linspace(0, last_time_step, new_time_step, endpoint=True))
        # plt.plot(np.linspace(0, last_time_step, new_time_step, endpoint=True), x)
        # plt.plot(range(particles_matrix[i, :, 0].shape[0]), particles_matrix[i, :, 0])
        #
        # plt.plot()
        # plt.show()

    return new_particles_matrix


def compute_ab_frames():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {u: int(v) for u, v in [du.split(" ")[:2] for du in lines2]}
    traj = []
    for frn in lines:
        akey = "1-%s.png" % frn
        traj.append(frame2ab[akey])
    for idx, num in enumerate(traj):
        if np.all([traj[idx+du] > 1000 for du in range(5)]):
            return idx, len(traj)-idx-1
    raise RuntimeError


def write_to_pcd(particles_matrix, save_folder='../data_heavy/sph_solutions/new_state/'):
    os.makedirs("../data_heavy/sph_solutions/new_state/", exist_ok=True)
    for i in tqdm(range(particles_matrix.shape[1]), desc="Writing new vertices"):
        p_in_time_step = particles_matrix[:, i, :]
        write_file = os.path.join(save_folder, f"new_particles_{i}.txt")
        with open(write_file, 'w') as f:
            for p in p_in_time_step:
                if (int(p[0]) != -100) and (int(p[1]) != -100) and (int(p[2]) != -100):
                    s = f'%f %f %f %f %f %f\n' % (p[0], p[1], p[2], 0, 0, 0)
                    f.write(s)
                else:
                    break
            f.close()
    return


def compute_ab_pose():
    sys.stdin = open("../data_heavy/frames/info.txt")
    lines = [du[:-1] for du in sys.stdin.readlines()]
    sys.stdin = open("../data_heavy/frame2ab.txt")
    lines2 = [du[:-1] for du in sys.stdin.readlines()]
    frame2ab = {du.split(" ")[0]: du for du in lines2}
    dist_all_x = []
    dist_all_y = []
    ab_area_all = []
    head_area_all = []
    rot_all = []
    scale_all = []
    for frn in lines:
        akey = "1-%s.png" % frn
        _, ab_area, head_area, dist_x, dist_y, _, rot = frame2ab[akey].split(" ")
        ab_area, head_area = map(int, [ab_area, head_area])
        if ab_area > 1000:
            ab_area_all.append(ab_area)
        if head_area > 1000:
            head_area_all.append(head_area)
        if ab_area > 1000:
            dist_x, dist_y, rot = map(int, [dist_x, dist_y, rot])
            dist_all_x.append(dist_x)
            dist_all_y.append(dist_y)
            rot_all.append(rot)
    abam1, ham1 = np.mean(ab_area_all), np.mean(head_area_all)
    for frn in lines:
        akey = "1-%s.png" % frn
        _, ab_area, head_area, dist_x, dist_y, _, rot = frame2ab[akey].split(" ")
        ab_area, head_area = map(int, [ab_area, head_area])
        if ab_area > abam1 and head_area > ham1:
            scale_all.append(head_area / ab_area)

    return np.mean(scale_all), np.mean(dist_all_x), np.mean(dist_all_y), np.mean(rot_all)


if __name__ == "__main__":
    # compute_ab_pose()
    start, nb_frames = compute_ab_frames()
    a = sample_accordingly(nb_frames)
    write_to_pcd(a)

