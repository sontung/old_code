import os
import shutil
import subprocess
import argparse
from distutils.dir_util import copy_tree
from tqdm import tqdm
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', type=bool, default=False, help='Debug mode')
parser.add_argument('-s', '--seg', type=int, default=0,
                    help='Index of model to run segmentation\n0: Swin\n1: DeepLab')

args = vars(parser.parse_args())
DEBUG_MODE = args['debug']
MODEL_INDEX = args['seg']

existing_models = (0, 1)


def clean_stuffs():
    print("Clean any stuffs you may have")
    shutil.rmtree('data_heavy', ignore_errors=True)
    shutil.rmtree('sph_data/mc_solutions', ignore_errors=True)
    shutil.rmtree('sph_data/mc_solutions_smoothed', ignore_errors=True)


class StringColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def move_video(folder_input, folder_output, dst_fd='data_const/run', result_df='data_const/final_vis',
               debug_mode=DEBUG_MODE):

    sub_folders = os.walk(folder_input).__next__()[1]
    sub_out_folders = os.walk(folder_output).__next__()[1]

    if len(sub_folders) >= 2:
        clean_stuffs()

    if MODEL_INDEX not in existing_models:
        print("Index of model not exist")
        return

    if MODEL_INDEX == 0:
        command = './run.sh'
    else:
        command = './run_with_deeplab.sh'

    sub_folders2 = [du for du in sub_folders if du not in sub_out_folders]
    print(f"skipping {len(sub_folders)-len(sub_folders2)} videos as they are already completed")
    for folder in tqdm(sub_folders2, desc=f"{StringColors.OKGREEN}Running all video{StringColors.ENDC}"):

        # delete all file in data_const/run
        for fd in glob(dst_fd + '/*'):
            os.remove(fd)

        print(f"\nvideo folder: {folder}")
        if folder in sub_out_folders:
            print("  skipping")
            continue
        src_fd = os.path.join(folder_input, folder)
        shutil.copytree(src_fd, dst_fd, dirs_exist_ok=True)

        # start run
        subprocess.call(command)

        if len(glob(f"{result_df}/*")) < 6:  # if produced less than 5 output files => run failed
            print(f"{folder} doesn't complete because it produced {len(glob(f'{result_df}/*'))}")
            if debug_mode:
                return
            else:
                clean_stuffs()
                continue

        # move result
        save_result = os.path.join(folder_output, folder)
        shutil.move(result_df, save_result)
        copy_tree("sph_data/mc_solutions_smoothed", f"{save_result}/mc_solutions_smoothed")

        # clean
        if not debug_mode:
            shutil.rmtree("data_const/final_vis", ignore_errors=True)
            clean_stuffs()


if __name__ == '__main__':
    os.makedirs('data_video/all_final_vis', exist_ok=True)
    move_video("data_video/all_video", "data_video/all_final_vis")

