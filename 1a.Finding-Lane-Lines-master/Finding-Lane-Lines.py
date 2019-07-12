# Import Package
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip


# Ideas for Lane Detection Pipeline

# Helper Functions
def get_roi(img): #region of interest
    #defining a blank mask to start with
    mask = np.zeros_like(img)   #建一個維度和img一樣的矩陣，初始化為0
    #openCV最左上角為原點
    # define vertices by four points
    xsize = img.shape[1]#x維大小,寬
    ysize = img.shape[0]#y維大小,高
    left_bottom = (80, ysize)#80x最下方 的點
    left_top = (xsize / 2 - 50, ysize / 2 + 50)#左下方的點
    right_bottom = (xsize - 80, ysize)#右邊數過來80x最下方 的點
    right_top = (xsize / 2 + 50, ysize / 2 + 50)#右上方的點
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)#建一個梯形區域的矩陣
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:#通道為3通道 RGB色
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image   通道數
        ignore_mask_color = (255,) * channel_count#(255,255,255) or 4個255 白色
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)#將矩陣內梯形設為白色
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)# 傳回遮罩區內的圖 用and
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)#img,開始點,結束位點,顏色,厚度
        
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)#無限延伸的直線
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)#建立一張img的白圖
    draw_lines(line_img, lines[0])#對line線裡面塗紅色，預設厚度10
    return line_img



# Test on Videos
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#將圖片轉成灰階
    blur = cv2.GaussianBlur(gray, (5, 5), 0)#高斯模糊，filter為5x5
    edges = cv2.Canny(blur, 50, 200)#尋找邊緣，閥值為50~200
    roi = get_roi(edges)#取得邊緣內的圖，只套用梯形內的區域(盲猜範圍，古老但好用 this method is old but efficient.)
    lines = hough_lines(roi, 2, np.pi / 180, 15, 5, 20)#用霍夫變換找出直線。輸入roi區域、距離解析度2、角度解析度pi/180、閥值15、線最短距離5、最大間隔20
    result = cv2.addWeighted(image, 0.8, lines, 1., 0.)#圖片融合，img權重0.8，line權重1，gamma=0
    return result


# solidWhiteRight.mp4
#white_output = 'test_videos_output/white.mp4'
#clips = VideoFileClip("test_videos/solidWhiteRight.mp4")#此為solidWhiteRight路徑(影片input)
#line_clips = clips.fl_image(process_image)  #process_image處理方式，fl_image filter濾鏡_圖片，對clips進行process_image一系列圖片處理
#line_clips.write_videofile(white_output, audio=False)

# solidYellowLeft
#yellow_output = 'test_videos_output/yellow.mp4'
#clips = VideoFileClip("test_videos/solidYellowLeft.mp4")
#line_clips = clips.fl_image(process_image) #證明黃色也能找，只是都找直線
#line_clips.write_videofile(yellow_output, audio=False)



# Improved lane finding pipeline

def divide_lines(img, lines):
    x_middle = img.shape[1] / 2#寬度一半 影片寬度為960，因此一半為480
    all_left_lines = []
    all_right_lines = []
    left_lines = []
    right_lines = []
#    print(lines)
    for line in lines:
        if abs(line[0][0] - line[0][2]) > 2:#寬相減大於2
            k = (line[0][3] - line[0][1]) * 1.0 / (line[0][2] - line[0][0])#計算斜率 dy/dx
            if line[0][0] < x_middle and k < -0.5:#點在左半，斜率小於-0.5
                all_left_lines.append(line[0])#左線加入line
            elif line[0][2] > x_middle and k > 0.5:#點在右半且斜率>0.5
                all_right_lines.append(line[0])#右線加入line
                
    all_left_lines.sort(key=lambda x: x[0])#依照x點排列
    all_right_lines.sort(key=lambda x: x[0])
    
    for line in all_left_lines:
        if len(left_lines) != 0:
            if line[0] > left_lines[-1][2] and line[1] < left_lines[-1][3]:# [-1]為最後一條線，若x點更靠右，y點更靠下，則
                left_lines.append([left_lines[-1][2], left_lines[-1][3], line[0], line[1]]) #增加最後一條線的點和現在點連成的線
                left_lines.append([line[0], line[1], line[2], line[3]])#增加現在的線
        else:
            left_lines.append([line[0], line[1], line[2], line[3]])#初始加入前兩個點，一條直線

    for line in all_right_lines:#同left
        if len(right_lines) != 0:
            if line[0] > right_lines[-1][2] and line[1] > right_lines[-1][3]:
                right_lines.append([right_lines[-1][2], right_lines[-1][3], line[0], line[1]])
                right_lines.append([line[0], line[1], line[2], line[3]])
        else:
            right_lines.append([line[0], line[1], line[2], line[3]])

    return left_lines, right_lines


def improved_lines(left_lines, right_lines, shape):
    ysize = shape[0]
    left_bottom = [left_lines[0][0], left_lines[0][1]]#左邊第一條線的點 最左下的點
    left_top = [left_lines[-1][2], left_lines[-1][3]]#左邊最後一條線的點 最右上的點
    right_top = [right_lines[0][0], right_lines[0][1]]#右第一條線的點 最右下點
    right_bottom = [right_lines[-1][2], right_lines[-1][3]]#右邊 最左上點

    k_left = (left_top[1] - left_bottom[1]) * 1.0 / (left_top[0] - left_bottom[0]) #左斜率 <0
    k_right = (right_top[1] - right_bottom[1]) * 1.0 / (right_top[0] - right_bottom[0])#右斜率

    left_bottom2 = [int(left_bottom[0] - (left_bottom[1] - ysize) / k_left), ysize]#將左下點延長到最左下邊緣
    left_top2 = [int(left_top[0] - (left_top[1] - (ysize / 2 + 50)) / k_left), int(ysize / 2 + 50)]#左半 右上點延伸到中間偏下處
    right_bottom2 = [int(right_bottom[0] - (right_bottom[1] - ysize) / k_right), ysize]
    right_top2 = [int(right_top[0] - (right_top[1] - (ysize / 2 + 50)) / k_right), int(ysize / 2 + 50)]
    
    left_lines.append([left_bottom[0], left_bottom[1], left_bottom2[0], left_bottom2[1]])#多加兩條線：左下延伸線、
    left_lines.append([left_top[0], left_top[1], left_top2[0], left_top2[1]])#右上延伸線
    right_lines.append([right_bottom[0], right_bottom[1], right_bottom2[0], right_bottom2[1]])
    right_lines.append([right_top[0], right_top[1], right_top2[0], right_top2[1]])
    
    print(left_lines)
    return left_lines, right_lines

def improved_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)#前面和舊版的一樣
    
    left_lines, right_lines = divide_lines(img, lines)#此為新加的，強化線條判斷，並分成兩半分顏色，增加更多的線
    left_lines, right_lines = improved_lines(left_lines, right_lines, img.shape)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, left_lines, [255, 0, 0], 10)
    draw_lines(line_img, right_lines, [0, 255, 0], 10)

    return line_img


def improved_process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)#轉灰色
    blur = cv2.GaussianBlur(gray, (5, 5), 0)#高斯模糊
    edges = cv2.Canny(blur, 50, 200)#取得邊緣
    roi = get_roi(edges)#取得中間區域
    lines = improved_hough_lines(roi, 2, np.pi / 180, 15, 5, 20)#畫線
    result = cv2.addWeighted(image, 0.8, lines, 1., 0.)#合併影像
    return result

#clips.close()
white_output_improved = 'test_videos_output/white-improved.mp4'
clips = VideoFileClip("test_videos/solidWhiteRight.mp4")
line_clips = clips.fl_image(improved_process_image) #使用加強的處理方式
line_clips.write_videofile(white_output_improved, audio=False)

#yellow_output_improved = 'test_videos_output/yellow-improved.mp4'
#clips = VideoFileClip('test_videos/solidYellowLeft.mp4')
#line_clips = clips.fl_image(improved_process_image)
#line_clips.write_videofile(yellow_output_improved, audio=False)


# Optional Challenge
#challenge_output = 'test_videos_output/challenge.mp4'
#clips = VideoFileClip('test_videos/challenge.mp4')
#line_clips = clips.fl_image(process_image)#用古老的方法
#line_clips.write_videofile(challenge_output, audio=False)

#clips.close()
# Optional Challenge
#challenge_output = 'test_videos_output/challenge-improved.mp4'
#clips = VideoFileClip('test_videos/challenge.mp4')
#line_clips = clips.fl_image(improved_process_image)#傑出的一手
#line_clips.write_videofile(challenge_output, audio=False)

