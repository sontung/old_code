from solve_position import b_spline_smooth
from scipy import interpolate
import numpy as np
import json
import glob


def sample_accordingly(nb_frames):
    all_json = glob.glob("../data_heavy/sph_solutions/state/*.json")
    all_json = sorted(all_json, key=lambda du: float(du.split("/")[-1].split("state_")[1].split("_particle")[0]))
    print(all_json)
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

if __name__ == "__main__":
    main()
