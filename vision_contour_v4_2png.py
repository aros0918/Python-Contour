import cv2
import numpy as np

import os
import time
import cv2
import sys

class mycontour:
    shape = 0
    color = (0,0,0)
    center = (0,0)
    location = []

# Set the directory to watch
directory = os.path.dirname(os.path.abspath(sys.argv[0]))
print("next file")
print(directory)
# Wait until the file appears in the directory
while True:
    if "input.mp4" in os.listdir(directory):
        break
    time.sleep(1)

# Open the video file
cap = cv2.VideoCapture(os.path.join(directory, "input.mp4"))
# cap = cv2.VideoCapture(os.path.join(directory, "luffy.mp4"))

# Create a "scene" folder if it doesn't exist
scene_directory = os.path.join(directory, "/scene")
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Created directory: {directory}")

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")


# Define the output video parameters
output_file = os.path.join(directory, "processed.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

all_video_array = []
no_frame = 1
#cap.set(cv2.CAP_PROP_POS_FRAMES, 26-1)
# Loop through each frame of the video
while cap.isOpened():
    frame_directory = os.path.join(directory+"/scene", f"frame_{no_frame}")

    if not os.path.exists(frame_directory):
        os.makedirs(frame_directory)
        # print(f"Created directory: {frame_directory}")

    # Read the current frame
    ret, frame = cap.read()
    
    # Check if the frame was successfully read
    if not ret:
        break

    frame = cv2.flip(frame, 0)

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges1 = cv2.Canny(gray, 50, 100)
    black = cv2.threshold(edges1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    black_contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ###---### Use canny edge + preprocessing to get more detail contours
    blur = cv2.blur(gray,(3,3))
    edges = cv2.Canny(blur, 100, 200)
    kernel = np.ones((3, 3), np.uint8)
    edges  = cv2.morphologyEx(edges , cv2.MORPH_CLOSE,kernel, iterations=1)
    border_size = 1
    edges = cv2.copyMakeBorder(
        edges,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 25]
        )
    #cv2.imshow('edges', edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank image with the same size as the original image
    output3 = np.zeros_like(frame)
    # Loop through each contour and find the average color of the pixels inside the contour
    for contour in contours:
        # Create a mask for the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

        # Create a mask for the layer inside the contour
        layer_mask = np.zeros_like(gray)
        cv2.drawContours(layer_mask, [contour], 0, 255, thickness=-1)

        # Find the average color of the pixels inside the layer mask
        average_color = np.mean(frame[np.where(layer_mask == 255)], axis=0)
        average_color = tuple(map(int, average_color))  # convert to integer tuple

        # Fill the contour with the average color on the output image
        cv2.fillPoly(output3, [contour], average_color)


    #cv2.drawContours(output3, black_contours, -1, 0, 2)

    mask_black = np.zeros_like(gray)
    cv2.drawContours(mask_black, black_contours, -1, 255, 2)
    output3[mask_black>0] = (0,0,0)

    ###---### Use canny edge + preprocessing to get more detail contours

    ###---### find contour again from "output3" to get separated contours
    gray3 = cv2.cvtColor(output3, cv2.COLOR_BGR2GRAY)
    _,th3 = cv2.threshold(gray3,0,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Create a blank image with the same size as the original image
    output4 = np.zeros_like(frame)

    myarray = []
    count = 1
    index = 0

    # Loop through each contour and find the average color of the pixels inside the contour
    for contour in contours:
        # Create a mask for the contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], 0, 255, cv2.FILLED)

        #check if contour is the black lines (from mask_black),if yes, skip it
        similarity = sum(mask[mask>0] + mask_black[mask>0])/len(mask_black[mask>0])
        Hit = cv2.bitwise_and(mask[mask>0], mask_black[mask>0] )
        similarity = np.count_nonzero(Hit)/len(mask_black[mask>0])
        if similarity > 0.5:
            continue

        # Create a mask for the layer inside the contour
        layer_mask = np.zeros_like(gray)
        cv2.drawContours(layer_mask, [contour], 0, 255, thickness=-1)
        kernel = np.ones((3, 3), np.uint8)
        #mask  = cv2.morphologyEx(mask , cv2.MORPH_DILATE,kernel, iterations=1)

        # Find the average color of the pixels inside the layer mask
        average_color = np.mean(frame[np.where(layer_mask == 255)], axis=0)
        average_color = tuple(map(int, average_color))  # convert to integer tuple

        # Fill the contour with the average color on the output image
        #cv2.fillPoly(output4, [contour], average_color)

        M = cv2.moments(contour)
        if M["m00"] == 0:
            cX = 0
            cY = 0
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        temp = mycontour()
        temp.shape = count
        temp.color = average_color
        temp.center = (cY,cX)
        temp.location = contour
        if len(contour)> 10 and len(contour)< 30:
            index = count
        myarray.append(temp)

        output4[mask>0] = average_color
        count = count + 1

    ### make the black lines as same as original
    ### Tone down the black lines using morphological erosion
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, black_contours, -1, 255, 3)
    kernel = np.ones((3, 3), np.uint8)
    #mask  = cv2.morphologyEx(mask , cv2.MORPH_ERODE,kernel, iterations=1)
    #output4[mask>0] = frame[mask>0,:]


    #cv2.imshow('mask',mask)
    #'''
    black_lines = []
    black_lines_color = []
    th_color = 255
    for w in range(gray.shape[1]):
        for h in range(gray.shape[0]):
            if mask[h,w] > 0 and not (frame[h,w,0] > th_color and frame[h,w,1] > th_color and frame[h,w,2] > th_color):
                output4[h,w,:] = frame[h,w,:] 
                black_lines.append((h,w))
                black_lines_color.append(frame[h,w,:])

    temp = mycontour()
    temp.shape = count
    temp.color = black_lines_color
    temp.center = (0,0)
    temp.location = black_lines
    myarray.append(temp)

    i = index
    # print('#contour=',i)
    # print('shape=',myarray[i].shape)
    # print('color=',myarray[i].color)
    # print('lencolor=',len(myarray[i].color))
    # print('center=',myarray[i].center)
    # print('location=',myarray[i].location)
    # print('number of contour=',count)

    height = frame.shape[1]
    width = frame.shape[0]

    mask = np.zeros((frame.shape),np.uint8)
    #mask = frame.copy()*0
    if len(myarray[cnt.shape].color) <= 3:
        #print('cnt.color=',cnt.color)
        #cv2.drawContours(mask, cnt.location, -1, cnt.color, cv2.FILLED)
        cv2.fillPoly(mask, [myarray[cnt.shape].location], cnt.color)
    else:
        for idx,loc in enumerate(myarray[cnt.shape].location):
            #print('color=',cnt.color[idx])
            #print('loc=',loc)
            #print('lenloc=',len(loc))
            if hasattr(loc,"__len__") and loc[1] < height and loc[0] < width:
                mask[loc[0],loc[1],:] = myarray[cnt.shape].color[idx]
                #print('loc=',loc[1],',',loc[0])                      
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0) 
    # filename = frame_directory + "contour_" + str(cnt.shape) + ".png"
    filename = os.path.join(frame_directory, f"contour_{cnt.shape}.png")
    frame_print = os.path.join(frame_directory, f"front.png")

    cv2.imwrite(filename, mask) 
    cv2.imwrite(frame_print, frame) 
        # print(filename)
        #cv2.imshow('mask',mask)
        #cv2.waitKey(0)  

    #'''  

    ###---### find contour again from "output3" to get separated contours


    #cv2.imshow('Result3', output3)
    # cv2.imshow('Result4', output4)

    # cv2.imshow('ori', frame)
    
    all_video_array.append(myarray)

    output = output4

    # cv2.waitKey(0)
    out.write(output)
    no_frame = no_frame + 1 
    # print("Directory:", directory)

    # Wait for the user to press a key to exit
    #if cv2.waitKey(25) & 0xFF == ord('q'):
        #break
# Release the video file and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()