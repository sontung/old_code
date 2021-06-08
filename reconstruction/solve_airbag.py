from solve_position import b_spline_smooth
import json
import glob

def main():
    all_json = glob.glob("../data_heavy/sph_solutions/state/*.json")
    all_json = sorted(all_json, key=lambda du: float(du.split("/")[-1].split("state_")[1].split("_particle")[0]))
    print(all_json)
    for file1 in all_json:
        with open(file1, "r") as fp:
            data = json.load(fp)
        particle_data = data["particles"]
        nb_particles = len(particle_data)
        for a_key in particle_data:
            particle = particle_data[a_key]
            print(particle["id"], particle["position"])
    return

if __name__ == "__main__":
    main()
