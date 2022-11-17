import numpy as np
import cv2
from scipy import signal

def gauss_dx(x, y, sigma):
    return -(x / (2 * np.pi * sigma ** 4)) * np.exp((- x ** 2 - y ** 2) / (2 * sigma * sigma))

def gauss_dy(x, y, sigma):
    return -(y / (2 * np.pi * sigma ** 4)) * np.exp((- x ** 2 - y ** 2) / (2 * sigma * sigma))

def gauss_dx_kernel(rad, sigma):
    kernel = np.array([[gauss_dx(i - rad, j - rad, sigma) for j in range(2 * rad + 1)] for i in range(2 * rad + 1)])
    return kernel

def gauss_dy_kernel(rad, sigma):
    kernel = np.array([[gauss_dy(i - rad, j - rad, sigma) for j in range(2 * rad + 1)] for i in range(2 * rad + 1)])
    return kernel

def grad_module(image, sigma):
    rad = int(np.ceil(3 * sigma))
    G_x = signal.convolve2d(image, gauss_dx_kernel(rad, sigma), mode='same', boundary='symm')
    G_y = signal.convolve2d(image, gauss_dy_kernel(rad, sigma), mode='same', boundary='symm')
    module = np.sqrt(G_x * G_x + G_y * G_y)
    angle = np.arctan2(G_y, G_x)
    return 255 * module / np.max(module), angle

def nonmax_suppression(image, sigma):
    grad, angle = grad_module(image, sigma)
    angle = (180./ np.pi) * angle
    angle[angle < 0] += 180
    res = np.zeros(grad.shape)
    grad = np.pad(grad, pad_width=1, mode='edge')
    for i in range(1, grad.shape[0] - 1):
        for j in range(1, grad.shape[1] - 1):
            q = 0
            p = 0
            if 0 <= angle[i - 1][j - 1] < 22.5  or 157.5 <= angle[i - 1][j - 1] <= 180:
                q = grad[i - 1][j]
                p = grad[i + 1][j]
                
            elif 22.5 <= angle[i - 1][j - 1] < 67.5:
                 q = grad[i + 1][j + 1]
                 p = grad[i - 1][j - 1]
                
            elif 67.5 <= angle[i - 1][j - 1] < 112.5:
                q = grad[i][j + 1]
                p = grad[i][j - 1]
                
            else:
                q = grad[i + 1][j - 1]
                p = grad[i - 1][j + 1] 
                          
            if grad[i][j] >= p and grad[i][j] >= q:
                res[i - 1][j - 1] = grad[i][j]         

    return res      

def hysteresis(nonmax, thr1, thr2):
    res = np.zeros(nonmax.shape)
    thr1 *= 255
    thr2 *= 255
    res_pad = np.pad(res, pad_width=1)
    weak = set()
    for i in range(1, res_pad.shape[0] - 1):
        for j in range(1, res_pad.shape[1] - 1):
            if nonmax[i - 1][j - 1] >= thr2:
                res_pad[i][j] = 255
            elif thr1 < nonmax[i - 1][j - 1] < thr2:
                weak.add((i, j))
    cnt = 1
    while(cnt > 0):
        cnt = 0
        weak_remove = set()
        for i, j in weak:
            if res_pad[i][j + 1] == 255 or res_pad[i + 1][j] == 255 or res_pad[i + 1][j + 1] == 255 \
                or res_pad[i - 1][j] == 255 or res_pad[i][j - 1] == 255 or res_pad[i - 1][j - 1] == 255 \
                    or res_pad[i - 1][j + 1] == 255 or res_pad[i + 1][j - 1] == 255:
                        res_pad[i][j] = 255
                        cnt += 1
                        weak_remove.add((i, j))
        weak.difference_update(weak_remove)
                        
    return res_pad[1:res_pad.shape[0] - 1, 1:res_pad.shape[1] - 1]

def canny(img, sigma, thr1, thr2):
    if thr1 >= thr2:
        thr1, thr2 = thr2, thr1
    nonmax = nonmax_suppression(img, sigma)
    return hysteresis(nonmax, thr1, thr2)