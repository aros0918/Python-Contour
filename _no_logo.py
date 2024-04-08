
import cv2
import os
import shutil
import numpy as np
import glob
import time
import argparse
from collections import deque


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--back", type=str, help="The background file.")
args = parser.parse_args()

def count_folders_with_prefix(prefix):
    current_directory = os.getcwd()
    folder_count = 0

    for item in os.listdir(current_directory):
        if os.path.isdir(item) and item.startswith(prefix):
            folder_count += 1

    return folder_count
prefix = "out_"
result_int = count_folders_with_prefix(prefix)
if os.path.exists('scene'):
    shutil.rmtree('scene')
os.mkdir('scene')

if os.path.exists(args.back):
    back = cv2.imread(args.back)
    back_int = 1
else:
    back_int = 2


dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
def is_valid_pixel(x, y, rows, cols):
    return 0 <= x < cols and 0 <= y < rows
def bfs(image, start_x, start_y):
        
    edge_points = []

    queue = deque()
    queue.append((start_x, start_y))
    visited[start_y, start_x] = 255

    while queue:
        x, y = queue.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]                
            if is_valid_pixel(nx, ny, rows, cols) and visited[ny, nx] < 100:
                visited[ny, nx] = 255
                
                if image[ny, nx] == 255:
                    edge_points.append((nx, ny))
                else:
                    queue.append((nx, ny))
    return edge_points
frame_number = 0
output_video_path = f'scene/output.mp4'
png_files = glob.glob("out_0/*.png")
img = cv2.imread(png_files[0])
fps = 120
video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (img.shape[1], img.shape[0]))


for k in range(result_int):
    png_files = glob.glob("out_" + str(k) + "/*.png") 
    png_files.sort(key=lambda x: x.split(".")[0])

    if not png_files:
        print(f"No PNG files found in out_{k} directory.")
        continue

    file_path = f"scene_{k}_animations/1.txt"
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
        if (index) % 6 == 0:
            png_file = 'out_' + str(k) + '/' + str(index) + '.png'
            print(png_file)
            img = cv2.imread(png_file)
            img_copy = img.copy()
            rows, cols = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            edges1 = cv2.Canny(gray, 50, 100)
            black = cv2.threshold(edges1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            black_contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_black = np.zeros_like(gray)
            mask = np.zeros_like(img_copy)
            cv2.drawContours(mask_black, black_contours, -1, (255,255,255), 2)
            visited = np.zeros((rows, cols), dtype=np.uint8)
                
            edge_points = bfs(mask_black, 0, 0)
            for i in range(rows):
                for j in range(cols):
                    if visited[j][i] == 255:
                        
                        img_copy[j, i] = (0,0,0)

            s = img_copy
            blurred = cv2.blur(s, (10, 10), dst=None, borderType=cv2.BORDER_DEFAULT)
            s = np.where(mask != 0, blurred, s)

            kernel = np.ones((10, 10), np.uint8)
            visited =  cv2.dilate(visited, kernel, iterations=1)
            visited =  cv2.erode(visited, kernel, iterations=1)
            if back_int == 1:
                result = back.copy()
                for i in range(rows):
                    for j in range(cols):
                        if visited[j][i] == 255:
                            s[j, i] = result[j, i]
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
            video.write(s)
        if (index + 1) % interval == 0:
            png_file = 'out_' + str(k) + '/' + str(index) + '.png'
            print(png_file)
            img = cv2.imread(png_file)
            img_copy = img.copy()
            rows, cols = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            edges1 = cv2.Canny(gray, 50, 100)
            black = cv2.threshold(edges1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            black_contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_black = np.zeros_like(gray)
            mask = np.zeros_like(img_copy)
            cv2.drawContours(mask_black, black_contours, -1, (255,255,255), 2)
            visited = np.zeros((rows, cols), dtype=np.uint8)
                
            edge_points = bfs(mask_black, 0, 0)
            for i in range(rows):
                for j in range(cols):
                    if visited[j][i] == 255:
                        
                        img_copy[j, i] = (0,0,0)

            s = img_copy
            blurred = cv2.blur(s, (10, 10), dst=None, borderType=cv2.BORDER_DEFAULT)
            s = np.where(mask != 0, blurred, s)

            kernel = np.ones((10, 10), np.uint8)
            visited =  cv2.dilate(visited, kernel, iterations=1)
            visited =  cv2.erode(visited, kernel, iterations=1)
            if back_int == 1:
                result = back.copy()
                for i in range(rows):
                    for j in range(cols):
                        if visited[j][i] == 255:
                            s[j, i] = result[j, i]
            if match_frame_count != frame_count:
                os.mkdir('scene/frame_' + str(frame_number))
                cv2.imwrite('scene/frame_' + str(frame_number) + '/frame.png', s)
                frame_number += 1
                match_frame_count += 1
       
        
video.release()

for i in range(result_int):
    shutil.rmtree(f'out_{i}')
    shutil.rmtree(f'scene_{i}_animations')

