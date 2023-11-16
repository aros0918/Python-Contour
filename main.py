
import cv2
from PIL import Image
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QPushButton
from collections import deque
def start():
    file1 = file_selector1.filename
    file2 = file_selector2.filename
    file3 = file_selector3.filename
    def imagebright(image):
        average_r = np.mean(image[:, :, 2])
        average_g = np.mean(image[:, :, 1])
        average_b = np.mean(image[:, :, 0])

        average_intensity = (average_r + average_g + average_b) / 3
        if average_intensity > 110:
            return True
        else:
            return False
    capVideo = cv2.VideoCapture(file1)
    fps = capVideo.get(cv2.CAP_PROP_FPS)
    frame_width = int(capVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
    t = cv2.imread(file2)
    t = cv2.resize(t, (frame_width, frame_height))
    background = cv2.imread(file3)
    background = cv2.resize(background, (frame_width, frame_height))
    t = cv2.cvtColor(t,cv2.COLOR_BGR2LAB)
    check = imagebright(t)

    def get_mean_and_std(x):
        x_mean, x_std = cv2.meanStdDev(x)
        x_mean = np.hstack(np.around(x_mean,2))
        x_std = np.hstack(np.around(x_std,2))
        return x_mean, x_std
        
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    def is_valid_pixel(x, y, rows, cols):
        return 0 <= x < cols and 0 <= y < rows

    # Perform BFS to obtain the edge points
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
    
    while True:
        result = background.copy()
        ret, videoFrame = capVideo.read()    
        if not ret:
            break
        # frame = videoFrame.copy()
        rows, cols = videoFrame.shape[:2]
        gray = cv2.cvtColor(videoFrame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("je", gray)
        height, width = gray.shape
        edges1 = cv2.Canny(gray, 50, 100)
        black = cv2.threshold(edges1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        black_contours, hierarchy = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask_black = np.zeros_like(gray)
        mask = np.zeros_like(videoFrame)
        cv2.drawContours(mask_black, black_contours, -1, (255,255,255), 2)
    
        visited = np.zeros((rows, cols), dtype=np.uint8)
        
        # kernel = np.ones((3, 3), np.uint8)
        
        # dilation = cv2.dilate(mask_black, kernel, iterations=1)
        # dilation = cv2.dilate(dilation, kernel, iterations=1)
        # dilation = cv2.dilate(dilation, kernel, iterations=1)
        # dilation = cv2.dilate(dilation, kernel, iterations=1)
        # dilation = cv2.dilate(dilation, kernel, iterations=1)
        # dilation = cv2.dilate(dilation, kernel, iterations=1)
        # erosion = cv2.erode(dilation, kernel, iterations=1)
        # erosion = cv2.erode(erosion, kernel, iterations=1)
        # erosion = cv2.erode(erosion, kernel, iterations=1)
        # erosion = cv2.erode(erosion, kernel, iterations=1)
        # mask_black = cv2.erode(erosion, kernel, iterations=1)
        
        edge_points = bfs(mask_black, 0, 0)
        for i in range(rows):
            for j in range(cols):
                if visited[j][i] == 255:
                    
                    videoFrame[j, i] = (0,0,0)

        s = videoFrame
        s = cv2.cvtColor(s,cv2.COLOR_BGR2LAB)
        
        s_mean, s_std = get_mean_and_std(s)
        t_mean, t_std = get_mean_and_std(t)
        height, width, channel = s.shape

        for i in range(0,height):
            for j in range(0,width):
                for k in range(0,channel):
                    x = s[i,j,k]
                    x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                    x = round(x)
                    x = 0 if x<0 else x
                    x = 255 if x>255 else x
                    s[i,j,k] = x

        s = cv2.cvtColor(s, cv2.COLOR_LAB2BGR)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                r, g, b = s[i, j]
                if check == False:
                    if r > 255/2:
                        r -= 40 * (r - 127)/ 100
                    else:
                        r += 40 * (r)/ 100
                    if g > 255/2:
                        g -= 40 * (g - 127)/ 100
                    else:
                        g += 40 * (g)/ 100
                    if b > 255/2:
                        b -= 20 * (b - 127)/ 100
                    else:
                        b += 20 * (b)/ 100
                    s[i, j] = [r, g, b]
                else:
                    s[i, j] = [r, g, b]

        for edge_point in edge_points:
            cv2.circle(mask, edge_point, radius=5, color=(255,255,255), thickness = -1)
        blurred = cv2.blur(s, (10, 10), dst=None, borderType=cv2.BORDER_DEFAULT)
        s = np.where(mask != 0, blurred, s)


        visiting = np.zeros((rows, cols), dtype=np.uint8)
        final = np.zeros((rows, cols), dtype=np.uint8)
        kernel = np.ones((10, 10), np.uint8)
        visited =  cv2.dilate(visited, kernel, iterations=1)
        visited =  cv2.erode(visited, kernel, iterations=1)
        # for i in range(rows):
        #     for j in range(cols):
        #         if visited[j][i] == 255:
        #             for k in range(16):
        #                 nx, ny = i + ddx[k], j + ddy[k]
        #                 if is_valid_pixel(nx, ny, rows, cols):
        #                     visiting[ny, nx] = 255
        # for i in range(rows):
        #     for j in range(cols):
        #         if visiting[j][i] != 255:
        #             for k in range(10):
        #                 nx, ny = i + dddx[k], j + dddy[k]
        #                 if is_valid_pixel(nx, ny, rows, cols):
        #                     final[ny, nx] = 255
        # for i in range(rows):
        #     for j in range(cols):
        #         if final[j][i] != 255:
        #             s[j, i] = result[j, i]
        for i in range(rows):
            for j in range(cols):
                if visited[j][i] == 255:
                    s[j, i] = result[j, i]
        # s = cv2.fastNlMeansDenoisingColored(s,None,20,20,7,21) 
        cv2.imshow('result', s)

        output_video.write(s)
        output_video.write(s)
        for j in range(2):
            capVideo.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capVideo.release()
    output_video.release()
    cv2.destroyAllWindows()

class FileSelector(QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        self.setLayout(vbox)

        button = QPushButton(self.title)
        button.clicked.connect(self.showDialog)
        vbox.addWidget(button)

        self.filename = ""

    def showDialog(self):
        options = QFileDialog.Options()
        options = QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, self.title, "", "All Files (*);;Python Files (*.py)", options=options)
        if filename:
            self.filename = filename

if __name__ == '__main__':
    app = QApplication([])
    file_selector1 = FileSelector("video file")
    file_selector2 = FileSelector("png file")
    file_selector3 = FileSelector("background file")
    # Create an additional button
    additional_button = QPushButton("start")
    additional_button.clicked.connect(start)
    # Add the button to the layout
    layout = QVBoxLayout()
    layout.addWidget(file_selector1)
    layout.addWidget(file_selector2)
    layout.addWidget(file_selector3)
    layout.addWidget(additional_button)
    
    # Create a main widget and set the layout
    main_widget = QWidget()
    main_widget.setLayout(layout)
    main_widget.show()

    app.exec_()