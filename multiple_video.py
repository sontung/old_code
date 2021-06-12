import os
import shutil
import subprocess


def move_video(folder_input, folder_output, dst_fd='data_const/run', result_df='data_const/final_vis'):
    sub_folders = os.walk(folder_input).__next__()[1]
    sub_out_folders = os.walk(folder_output).__next__()[1]
    for folder in sub_folders:
        print(folder)
        if folder in sub_out_folders:
            continue
        src_fd = os.path.join(folder_input, folder)
        shutil.copytree(src_fd, dst_fd, dirs_exist_ok=True)

        # start run
        subprocess.call('./run.sh')

        # move result
        save_result = os.path.join(folder_output, folder)
        shutil.move(result_df, save_result)

    return

move_video("/media/hblab/01D5F2DD5173DEA0/AirBag/all_videos",
           "/media/hblab/01D5F2DD5173DEA0/AirBag/all_final_vis")