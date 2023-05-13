import utils
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline, interp1d
import sys
import argparse
from grad_module import grad_module

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Path to input image')
    parser.add_argument('initial_snake', type=str, help='Path to .txt file with initial snake')
    parser.add_argument('output_path', type=str, help='Output image path')
    parser.add_argument('alpha', type=float, help='Alpha parameter')
    parser.add_argument('beta', type=float, help='Beta parameter')
    parser.add_argument('tau', type=float, help='Tau parameter')
    parser.add_argument('w_line', type=float, help='W_line parameter')
    parser.add_argument('w_edge', type=float, help='W_edge parameter')
    parser.add_argument('kappa', type=float, help='Balloon force parameter')
    return parser

def create_method_matrix(a, b, n):
    #create matrix for numerical solving of contour evolution equation
    A = 2 * a * np.eye(n) + 6 * b * np.eye(n)
    diag_1 = (-a - 4*b) * np.eye(n)
    A = A + np.roll(diag_1, 1, axis=1) + np.roll(diag_1, -1, axis=1)
    diag_2 = b * np.eye(n)
    A = A + np.roll(diag_2, 2, axis=1) + np.roll(diag_2, 2, axis=1)
    return A

def reparametrize_contour(contour, pow=1):
    #reparametrize our contour using chord length method
    new_contour = np.empty(contour.shape)
    s = []
    s.append(0)
    for i in range(1, contour.shape[0]):
        s.append(s[-1] + np.linalg.norm(contour[i, :] - contour[i - 1, :]))
    s.append(s[-1] + np.linalg.norm(contour[0, :] - contour[-1, :]))
    s = np.array(s)
    s /= s[-1]
    x = contour[:, 0]
    x_first = x[0]
    x = np.append(x, x_first)
    y = contour[:, 1]
    y_first = y[0]
    y = np.append(y, y_first)
    f_x = interp1d(s, x, kind='quadratic')
    f_y = interp1d(s, y, kind='quadratic')
    new_grid = np.linspace(0, 1, x.shape[0]) # there x.shape[0] is x initial length + 1
    new_contour[:, 0] = f_x(new_grid)[:-1]
    new_contour[:, 1] = f_y(new_grid)[:-1]
    unit_normal_vectors = np.zeros(new_contour.shape)
    dx = np.zeros(new_contour.shape[0])
    dy = np.zeros(new_contour.shape[0])
    h = 1 / (x.shape[0] - 1)
    n = new_contour.shape[0]
    for i in range(n):
        dx[i] = (-3 * new_contour[i % n, 0] + 4 * new_contour[(i + 1) % n, 0] - new_contour[(i + 2) % n, 0]) / (2 * h)
        dy[i] = (-3 * new_contour[i % n, 1] + 4 * new_contour[(i + 1) % n, 1] - new_contour[(i + 2) % n, 1]) / (2 * h)
    unit_normal_vectors[:, 0] = -dy
    unit_normal_vectors[:, 1] = dx
    unit_normal_vectors /= np.linalg.norm(unit_normal_vectors, axis=1).reshape((unit_normal_vectors.shape[0], 1))
    return new_contour, unit_normal_vectors

def contour_segmentation(image, init_contour, alpha, beta, tau, w_line, w_edge, kappa=0, n_iter=600, convergence=0.1, sigma=1):
    #image segmentation using active contours
    rad = int(np.ceil(3 * sigma))
    image_lines = cv2.GaussianBlur(image, (2 * rad + 1, 2 * rad + 1), sigma)
    image_edges = grad_module(image, sigma)
    potential = -w_line * image_lines - w_edge * image_edges
    f_ext = -potential

    interpolated_f_ext = RectBivariateSpline(np.arange(f_ext.shape[1]), np.arange(f_ext.shape[0]), f_ext.T)
    A = create_method_matrix(alpha, beta, init_contour.shape[0])
    A_inv = np.linalg.inv(np.eye(A.shape[0]) + tau * A)

    prev_snake = init_contour
    curr_snake = init_contour
    f_ext_new = np.empty(init_contour.shape)
    unit_normal_vectors = np.zeros(init_contour.shape)
    convergence_order = 10
    xsave = np.empty((convergence_order, init_contour.shape[0]), dtype=float)
    ysave = np.empty((convergence_order, init_contour.shape[0]), dtype=float)
    for i in range(n_iter):
        f_ext_new_x = interpolated_f_ext.ev(prev_snake[:, 0], prev_snake[:, 1], dx=1)
        f_ext_new_y = interpolated_f_ext.ev(prev_snake[:, 0], prev_snake[:, 1], dy=1)
        f_ext_new[:, 0] = f_ext_new_x
        f_ext_new[:, 1] = f_ext_new_y
        f_ext_new /= np.linalg.norm(f_ext_new, axis=1).reshape((f_ext_new.shape[0], 1))

        curr_snake = A_inv @ (prev_snake + tau * f_ext_new + kappa * unit_normal_vectors)
        x = curr_snake[:, 0]
        y = curr_snake[:, 1]
        x[x < 0] = 0
        y[y < 0] = 0
        x[x >= image.shape[0]] = image.shape[0] - 1
        y[y >= image.shape[1]] = image.shape[1] - 1
        curr_snake[:, 0] = x
        curr_snake[:, 1] = y 
        
        curr_snake, unit_normal_vectors = reparametrize_contour(curr_snake)

        x = curr_snake[:, 0]
        y = curr_snake[:, 1]
        j = i % (convergence_order + 1)
        if j < convergence_order:
            xsave[j, :] = x
            ysave[j, :] = y
        else:
            dist = np.min(np.max(np.abs(xsave - x[None, :])
                                 + np.abs(ysave - y[None, :]), 1))
            if dist < convergence:
                break
        #for testing
        #if i > 0 and i % 10 == 0:
            #utils.display_snake(image, init_contour, curr_snake, save=True, save_path=f"./test_iteration_{i}.png")
        prev_snake = curr_snake
    
    return curr_snake

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args(sys.argv[1:])

    image = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE)
    init_contour = np.loadtxt(args.initial_snake)
    result_snake = contour_segmentation(image, init_contour, args.alpha, args.beta, args.tau, args.w_line, args.w_edge, kappa=args.kappa, n_iter=600, convergence=0.2)
    utils.save_mask(args.output_path, result_snake, image)
    #for testing
    true_mask = cv2.imread('./task2_testdata/astranaut_mask.png', cv2.IMREAD_GRAYSCALE)
    pred_mask = cv2.imread(args.output_path, cv2.IMREAD_GRAYSCALE)
    print(f'IoU: {utils.iou(true_mask, pred_mask)}')