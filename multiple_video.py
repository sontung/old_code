import os
import shutil
import subprocess
from tqdm import tqdm
import sys


def move_video(folder_input, folder_output, dst_fd='data_const/run', result_df='data_const/final_vis'):
    sub_folders = os.walk(folder_input).__next__()[1]
    sub_out_folders = os.walk(folder_output).__next__()[1]
    for folder in tqdm(sub_folders, desc="Running all video"):
        print(f"\nvideo folder: {folder}")
        if folder in sub_out_folders:
            print("  skipping")
            continue
        src_fd = os.path.join(folder_input, folder)
        shutil.copytree(src_fd, dst_fd, dirs_exist_ok=True)

        # start run
        subprocess.call('./run.sh')

        # move result
        try:
            save_result = os.path.join(folder_output, folder)
            shutil.move(result_df, save_result)
        except:
            print(f"{folder} don't complete")

    return


os.makedirs('data_video/all_final_vis', exist_ok=True)
move_video("data_video/all_video", "data_video/all_final_vis")
