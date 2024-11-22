import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


videofile_tpl = ".../VideoMME/all_videos/*.mp4"
results_dir = ".../VideoMME/video_25_fps"
num_workers = 32
fps = 25

# For each video */videoname.mp4, this script will create a folder {results_dir}/videoname containing the extracted frames from the video.

### DO NOT CHANGE ###
overwrite_existing_results = True

def process_video(videofile):
    videoframe_dir = os.path.join(results_dir, videofile.split('/')[-1].replace(".mp4", ""))
    if os.path.exists(videoframe_dir):
        assert overwrite_existing_results
        os.system(f'rm -rf {videoframe_dir}')
    os.makedirs(videoframe_dir)
    framefile_tpl = os.path.join(videoframe_dir, "%06d.jpg")
    print(f"{videofile} -> {framefile_tpl}")
    os.system(f'ffmpeg -i {videofile} -vf fps={fps} {framefile_tpl} -hide_banner -loglevel error')

videofiles = glob.glob(videofile_tpl)

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    list(tqdm(executor.map(process_video, videofiles), total=len(videofiles)))