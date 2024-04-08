
import cv2
import os
import shutil
import numpy as np
import glob
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--logo", type=str, help="The logo png file.")
args = parser.parse_args()

if os.path.exists('scene'):
    shutil.rmtree('scene')
os.mkdir('scene')

def count_folders_with_prefix(prefix):
    current_directory = os.getcwd()
    folder_count = 0

    for item in os.listdir(current_directory):
        if os.path.isdir(item) and item.startswith(prefix):
            folder_count += 1

    return folder_count

prefix = "out_"
result = count_folders_with_prefix(prefix)

if os.path.exists(args.logo):
    logo = cv2.imread(args.logo)
    logo = cv2.resize(logo, (logo.shape[1] // 3, logo.shape[0] // 3)) 
    non_white_pixels = np.where((logo[:, :, :3] != [255, 255, 255]).any(axis=2))
    logo_int = 1
else:
    logo_int = 2

frame_number = 0
output_video_path = f'scene/output.mp4'
png_files = glob.glob("out_0/*.png")
img = cv2.imread(png_files[0])
fps = 120
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.shape[1], img.shape[0]))
for i in range(result):
    png_files = glob.glob("out_" + str(i) + "/*.png") 
    png_files.sort(key=lambda x: x.split(".")[0])

    if not png_files:
        print(f"No PNG files found in out_{i} directory.")
        continue

    file_path = f"scene_{i}_animations/1.txt"
    with open(file_path, 'r') as file:
        content = file.read()
        parts = content.split()
        duration = float(parts[0])
        frame_count = int(parts[1])

    if frame_count != 0:
        interval = len(png_files) // frame_count
    else:
        interval = 100000
    match_frame_count = 0
    if interval == 0:
        interval = 1     
    for index in range(len(png_files)):
        png_file = 'out_' + str(i) + '/' + str(index) + '.png'
        print(png_file)
        img = cv2.imread(png_file)
        if (index + 1) % interval == 0:
            if match_frame_count != frame_count:
                os.mkdir('scene/frame_' + str(frame_number))
                cv2.imwrite('scene/frame_' + str(frame_number) + '/frame.png', img)
                frame_number += 1
                match_frame_count += 1
        if logo_int == 1:
            img[-logo.shape[0]-30:, -logo.shape[1]-30:][non_white_pixels] = logo[non_white_pixels]
        video.write(img)
        video.write(img)
video.release()
for i in range(result):
    shutil.rmtree(f'out_{i}')
    shutil.rmtree(f'scene_{i}_animations')
directories = os.listdir()
for directory in directories:
    if "animations" in directory:
        os.remove(directory)


