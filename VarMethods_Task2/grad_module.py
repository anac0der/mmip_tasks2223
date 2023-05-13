from scipy import signal
import numpy as np

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
    return 255 * module / np.max(module)