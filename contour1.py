
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
    shutil.mkdir('scene')


if os.path.exists(args.logo):
    logo = cv2.imread(args.logo)
    logo = cv2.resize(logo, (logo.shape[1] // 3, logo.shape[0] // 3)) 
    non_white_pixels = np.where((logo[:, :, :3] != [255, 255, 255]).any(axis=2))
    logo_int = 1
else:
    logo_int = 2

output_video_path = f'scene/output.mp4'
png_files = glob.glob("out/*.png")
img = cv2.imread(png_files[0])
fps = 120
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.shape[1], img.shape[0]))
png_files.sort(key=lambda x: x.split(".")[0])

with open('frame_count.txt') as file:
    content = file.read()
    parts = content.split()
    frame_count = int(parts[0])
print(frame_count)
interval = int(len(png_files) / frame_count)
print(interval)
if len(png_files) < frame_count:
    interval = 1
folder_number = 0
for index in range(len(png_files)):
    png_file = 'out/' + str(index) + '.png'
    print(png_file)
    img = cv2.imread(png_file)
    img[-logo.shape[0]-30:, -logo.shape[1]-30:][non_white_pixels] = logo[non_white_pixels]

    if index % interval == 0:
        if folder_number != frame_count:
            os.mkdir('scene/frame_' + str(folder_number))
            cv2.imwrite('scene/frame_' + str(folder_number) + '/frame.png', img)
            folder_number += 1
            
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
    video.write(img)
video.release()
shutil.rmtree(f'out')
directories = os.listdir()
for directory in directories:
    if "animations" in directory:
        shutil.rmtree(directory)

