import os
import shutil


def remove_files_in_directory(folder):
    shutil.rmtree(folder, ignore_errors=True)
    return


if __name__ == "__main__":
    sph_solutions = "data_heavy"
    mc_solutions = 'sph_data/mc_solutions'
    mc_solutions_smoothed = 'sph_data/mc_solutions_smoothed'

    list_path_remove = [sph_solutions, mc_solutions, mc_solutions_smoothed]
    for _folder in list_path_remove:
        remove_files_in_directory(_folder)
