import os
import shutil
import subprocess
from tqdm import tqdm
from glob import glob


def move_video(folder_input, folder_output, dst_fd='data_const/run', result_df='data_const/final_vis'):
    sub_folders = os.walk(folder_input).__next__()[1]
    sub_out_folders = os.walk(folder_output).__next__()[1]
    for folder in tqdm(sub_folders, desc="Running all video"):
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
        subprocess.call('./run.sh')

        # move result
        try:
            save_result = os.path.join(folder_output, folder)
            shutil.move(result_df, save_result)
        except:
            print(f"{folder} doesn't complete")
            return
            shutil.rmtree('data_heavy', ignore_errors=True)
            shutil.rmtree('sph_data/mc_solutions', ignore_errors=True)
            shutil.rmtree('sph_data/mc_solutions_smoothed', ignore_errors=True)

    return


if __name__ == '__main__':
    os.makedirs('data_video/all_final_vis', exist_ok=True)
    move_video("data_video/all_video", "data_video/all_final_vis")
