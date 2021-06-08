import os.path

from solve_position import b_spline_smooth
from scipy import interpolate
import numpy as np
import json
import glob


def sample_accordingly3(new_time_step=42):
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
    last_time_step = nb_time_step-1
    # sampling frame
    for i in range(max_nb_particles):
        p = particles_matrix[i]
        # print(p[:, 0])
        null_idx = np.argwhere(p[:, 0] == -100)[:, 0]
        start_idx = 0
        if len(null_idx) != 0:
            start_idx = np.max(null_idx)
        tck_x = b_spline_smooth(p[start_idx:, 0], return_params=True)
        tck_y = b_spline_smooth(p[start_idx:, 1], return_params=True)
        tck_z = b_spline_smooth(p[start_idx:, 2], return_params=True)

        x = [interpolate.splev(du, tck_x) for du in np.linspace(0, last_time_step, new_time_step-start_idx, endpoint=True)]
        y = [interpolate.splev(du, tck_y) for du in np.linspace(0, last_time_step, new_time_step-start_idx, endpoint=True)]
        z = [interpolate.splev(du, tck_z) for du in np.linspace(0, last_time_step, new_time_step-start_idx, endpoint=True)]
        print(len(x), new_particles_matrix.shape, p[start_idx:, 0].shape)
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


def sample_accordingly(nb_frames):
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
                all_particles_pos[particle_id] = np.full((nb_time_step, 3), -1, dtype=float)
                all_particles_pos[particle_id][i] = particle["position"]
            else:
                all_particles_pos[particle_id][i] = particle["position"]

    particles_matrix = np.full((max_nb_particles, nb_time_step, 3), -1, dtype=float)
    for i in range(max_nb_particles):
        pos = all_particles_pos[i]
        particles_matrix[i][:len(pos)] = pos
    sampled_mat = np.zeros((max_nb_particles, nb_frames, 3))
    step = (nb_time_step-1)/(nb_frames-1)
    for xyz in range(3):
        for pid in range(max_nb_particles):
            f = b_spline_smooth(particles_matrix[pid][:, xyz], return_params=True)
            for ind in range(nb_frames):
                sampled_mat[pid][ind][xyz] = interpolate.splev(ind*step, f)
            
    return sampled_mat


def write_to_pcd(particles_matrix, save_folder='../data_heavy/sph_solutions/new_state/'):

    for i in range(particles_matrix.shape[1]):
        p_in_time_step = particles_matrix[:, i, :]
        write_file = os.path.join(save_folder, f"new_particles_{i}.txt")
        with open(write_file, 'w') as f:
            for p in p_in_time_step:
                if (p[0] != -100.0) and (p[1] != -100.0) and (p[2] != -100.0):
                    s = f'%s %s %s %s %s %s\n' % (p[0], p[1], p[2], 0, 0, 0)
                    f.write(s)
                else:
                    break
            f.close()
    return


if __name__ == "__main__":
    a = sample_accordingly3()
    # b = sample_accordingly(30)
    write_to_pcd(a)
    # print(a.shape, b.shape)
    # print(a[5500])
    # print(b[5500])
    # assert a == b

