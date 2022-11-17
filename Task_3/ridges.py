import numpy as np
import cv2
from scipy import signal
from canny import gauss_dx_kernel, gauss_dy_kernel

def compute_hessian(image, sigma):
    rad = int(np.ceil(3 * sigma))
    bound = 'symm'
    mode_c = 'same'
    G_xx, G_xy = np.gradient(gauss_dx_kernel(rad, sigma))
    _, G_yy = np.gradient(gauss_dy_kernel(rad, sigma))
    I_xx = signal.convolve2d(image, G_xx, mode=mode_c, boundary=bound)   
    I_xy = signal.convolve2d(image, G_xy, mode=mode_c, boundary=bound)   
    I_yy = signal.convolve2d(image, G_yy, mode=mode_c, boundary=bound)   
    return I_xx, I_xy, I_yy

def get_eigs(I_xx, I_xy, I_yy):
     res = np.zeros(I_xx.shape)
     res_vec = np.empty((I_xx.shape[0], I_xx.shape[1]), dtype='object')
     for i in range(I_xx.shape[0]):
        for j in range(I_xx.shape[1]):
            hessian = np.array([[I_xx[i][j], I_xy[i][j]], [I_xy[i][j], I_yy[i][j]]])
            eigs, eigvectors = np.linalg.eigh(hessian)
            max_eig_ind = 0
            if abs(eigs[1]) > abs(eigs[0]):
                max_eig_ind = 1
            if eigs[max_eig_ind] >= 0:
                res[i][j] = eigs[max_eig_ind]
            res_vec[i][j] = eigvectors[:, max_eig_ind]

     return res, res_vec

def nonmax_suppression_ridge(image, sigma):
    I_xx, I_xy, I_yy = compute_hessian(image, sigma)
    grad, vectors = get_eigs(I_xx, I_xy, I_yy)
    res = np.zeros(grad.shape)
    grad1 = np.pad(grad, pad_width=1, mode='edge')
    for i in range(1, grad1.shape[0] - 1):
        for j in range(1, grad1.shape[1] - 1):
            eigvector = vectors[i - 1][j - 1]         
            eigvector = np.around(eigvector).astype("int")
            p = grad1[i + eigvector[0]][j + eigvector[1]]
            q = grad1[i - eigvector[0]][j - eigvector[1]]
                        
            if grad1[i][j] > p and grad1[i][j] > q:
                res[i - 1][j - 1] = grad1[i][j] 
    return (255 / np.max(res)) * res

def hysteresis_ridge(nonmax, thr1, thr2):
    res = np.zeros(nonmax.shape)
    mapp = np.zeros(nonmax.shape)
    thr1 *= 255
    thr2 *= 255
    res_pad = np.pad(res, pad_width=1)
    mapp_pad = np.pad(mapp, pad_width=1)
    weak = set()
    for i in range(1, res_pad.shape[0] - 1):
        for j in range(1, res_pad.shape[1] - 1):
            if nonmax[i - 1][j - 1] >= thr2:
                res_pad[i][j] = nonmax[i - 1][j - 1]
                mapp_pad[i][j] = 255
            elif thr1 < nonmax[i - 1][j - 1] < thr2:
                weak.add((i, j))
    cnt = 1
    while cnt > 0:
        cnt = 0
        weak_remove = set()
        for i, j in weak:
            if mapp_pad[i][j + 1] == 255 or mapp_pad[i + 1][j] == 255 or mapp_pad[i + 1][j + 1] == 255 \
                or mapp_pad[i - 1][j] == 255 or mapp_pad[i][j - 1] == 255 or mapp_pad[i - 1][j - 1] == 255 \
                    or mapp_pad[i - 1][j + 1] == 255 or mapp_pad[i + 1][j - 1] == 255:
                        res_pad[i][j] = nonmax[i - 1][j - 1]
                        mapp_pad[i][j] = 255
                        cnt += 1
                        weak_remove.add((i, j))
        weak.difference_update(weak_remove)
                        
    return res_pad[1:res_pad.shape[0] - 1, 1:res_pad.shape[1] - 1]

def ridge_detection(image, s):
   if len(s) == 0:
   	   raise Exception("Пустой массив параметров!")
   thr1 = 0.05
   thr2 = 0.15
   res = np.zeros(image.shape)
   for sigma in s:
       cur_img = nonmax_suppression_ridge(image, sigma)
       cur_img = hysteresis_ridge(cur_img, thr1, thr2)
       res = np.maximum(res, cur_img)
   return res