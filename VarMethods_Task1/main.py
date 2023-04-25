import numpy as np
import cv2
from scipy import signal
import sys

def gradient(z, u, kernel, alpha, noise=0):
    temp = signal.convolve2d(z, kernel, mode='same', boundary='symm') - u + noise
    kernel_T = np.rot90(kernel, k=2)
    residual_part = (2 * signal.convolve2d(temp, kernel_T, mode='same', boundary='symm')).astype('float64')
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 or j != 0:
                grad = np.sign(np.roll(z, (i, j), axis=(1, 0)) - z)
                grad = np.roll(grad, (-i, -j), axis=(1, 0)) - grad
                grad = grad.astype('float64') / ((i ** 2 + j ** 2) ** 0.5)
                residual_part += alpha * grad
    
    return residual_part

def deconvolution(blurred_image, kernel, noise_level, alpha=0.1, gamma=0.85, initial_lr=1, n_iter=100):
    result = blurred_image.copy()
    for i in range(1, n_iter + 1):
        if i != 1:
            grad = gradient(result - gamma * v, blurred_image, kernel, alpha)
            v = gamma * v + initial_lr * np.exp(-(i / 100)) * grad
        else:
            grad = gradient(result, blurred_image, kernel, alpha)
            v = initial_lr * np.exp(-(i / 100)) * grad
        result -= v
    return result

input_path = sys.argv[1]
kernel_path = sys.argv[2]
output_path = sys.argv[3]
noise_level = float(sys.argv[4])

blurred_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE).astype('float64')
kernel = cv2.imread(kernel_path, cv2.IMREAD_GRAYSCALE).astype('float64')
kernel /= np.sum(kernel)

result = deconvolution(blurred_image, kernel, noise_level, alpha=max(0.1 * noise_level, 0.1))

cv2.imwrite(output_path, result)