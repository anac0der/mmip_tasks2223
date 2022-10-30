##Импорт необходимых библиотек
import numpy as np
import cv2

"""## Метрики

###MSE
"""

def MSE(image1, image2):
    if image1.shape != image2.shape:
        raise Exception("Размер изображений должен быть одинаковым!")
    return (1 / (image1.shape[0] * image1.shape[1])) * np.sum((image1 - image2) ** 2)

"""### PSNR"""

def PSNR(image1, image2, max_pixel_val):
    if image1.shape != image2.shape:
        raise Exception("Размер изображений должен быть одинаковым!")
    eps = 10 ** (-10)
    mse = MSE(image1, image2)
    return 10 * np.log10((max_pixel_val ** 2) / (mse + eps))

"""###SSIM"""

def SSIM(image1, image2, L):
    if image1.shape != image2.shape:
        raise Exception("Размер изображений должен быть одинаковым!")

    cov = np.cov(image1.flatten(), image2.flatten())
    mean1 = np.mean(image1)
    mean2 = np.mean(image2)
    var1 = cov[0][0]
    var2 = cov[1][1]
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    cov_num = cov[0][1]
    return ((2 * mean1 * mean2 + c1) * (2 * cov_num + c2)) / ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))

def MSSIM(image1, image2, window_size, L):
    if image1.shape != image2.shape:
        raise Exception("Размер изображений должен быть одинаковым!")
    if image1.shape[0] < window_size:
        raise Exception("Слишком большой размер окна!")

    windows_metric = np.array([])
    for i in range(image1.shape[0] - window_size):
        for j in range(image1.shape[0] - window_size):
            window1 = image1[i:i + window_size, j:j + window_size]
            window2 = image2[i:i + window_size, j:j + window_size]
            windows_metric = np.append(windows_metric, SSIM(window1, window2, L))
    return np.mean(windows_metric)

"""## Медианная фильтрация"""

def image_extension(image, rad):
    image1 = np.zeros((image.shape[0] + 2 * rad, image.shape[1] + 2 * rad))
    image1[rad:-rad, :rad] = (image1[rad:-rad, :rad].T + image[:, 0].T).T
    image1[rad:-rad, -rad:] = (image1[rad:-rad, -rad:].T + image[:, -1].T).T
    image1[:rad, rad:-rad] += image[0, :]
    image1[-rad:, rad:-rad] += image[-1, :]
    image1[:rad, :rad] += image[0][0]
    image1[:rad, -rad:] += image[0][-1]
    image1[-rad:, -rad:] += image[-1][-1]
    image1[-rad:, :rad] += image[-1][0]
    image1[rad:-rad, rad:-rad] = image
    return image1
    
def median_filter(image, rad):
    new_image = np.empty(image.shape)
    image1 = image_extension(image, rad)
    for i in range(rad, image1.shape[0] - rad):
        for j in range(rad, image1.shape[1] - rad):
            window_pixels = image1[i - rad : i + rad + 1, j - rad : j + rad + 1]
            new_image[i - rad][j - rad] = np.median(window_pixels)
    return new_image

"""## Билатеральная фильтрация"""

def w(image, i, j, k, l, sigma_d, sigma_r):
    return np.exp(-((i - k) ** 2 + (j - l) ** 2)/(2 * sigma_d * sigma_d) - (((image[i][j] - image[k][l]) ** 2)/(2 * sigma_r * sigma_r)))

def bilateral_filter(image, sigma_d, sigma_r):
    rad = int(min(3 * np.ceil(sigma_d), 5))
    new_image = np.empty(image.shape)
    image1 = image_extension(image, rad)
    for i in range(rad, image1.shape[0] - rad):
        for j in range(rad, image1.shape[1] - rad):         
            I = image1[i - rad : i + rad + 1, j - rad : j + rad + 1]
            W = np.array([[w(image1, i, j, k, l, sigma_d, sigma_r) for l in range(j - rad, j + rad + 1)] for k in range(i - rad, i + rad + 1)])
            new_image[i - rad][j - rad] = np.sum(I * W) / np.sum(W)
    return new_image

"""## Фильтр Гаусса"""

def gauss(x, y, sigma):
    return 1 / (2 * np.pi * sigma * sigma) * np.exp((- x ** 2 - y ** 2) / (2 * sigma * sigma))

def gauss_kernel(rad, sigma):
    kernel = np.array([[gauss(i - rad, j - rad, sigma) for j in range(2 * rad + 1)] for i in range(2 * rad + 1)])
    return kernel / np.sum(kernel)

def gauss_filter(image, sigma_d):
    rad = int(3 * np.ceil(sigma_d))
    new_image = np.empty(image.shape)
    kernel = gauss_kernel(rad, sigma_d)
    image1 = image_extension(image, rad)
    for i in range(rad, image1.shape[0] - rad):
        for j in range(rad, image1.shape[1] - rad):
            I = image1[i - rad : i + rad + 1, j - rad : j + rad + 1]
            new_image[i - rad][j - rad] = np.sum(I * kernel)
    return new_image

"""## Определение сдвига и поворота изображения"""

import skimage.transform

def autocontrast(image):
    return (255 / (np.max(image) - np.min(image))) * (image - np.min(image))

def logging(image):
    return autocontrast(np.log(1 + image))

def image_transform(image):
    mask = np.array([[-1 if (i + j) % 2 == 1 else 1 for j in range(image.shape[1])] for i in range(image.shape[0])])

    fft = np.abs(np.fft.fft2(image * mask))
    fft = gauss_filter(fft, 1)

    polar = skimage.transform.warp_polar(fft, output_shape = fft.shape, radius = min(fft.shape[0] / 2, fft.shape[1] / 2))
    fft = np.abs(np.fft.fft(polar * mask, axis=0))
    return fft

def detect_shift_and_rotate(img1, img2):
    tr_img1 = image_transform(img1)
    tr_img2 = image_transform(img2)
    return SSIM(logging(tr_img1), logging(tr_img2), 255)

"""##Основная часть программы"""

import sys
command = sys.argv[1]
result_img = None
output_path = None

if command == 'mse':
    path1 = sys.argv[2]
    path2 = sys.argv[3]
    img1 = np.array(cv2.imread(path1, cv2.IMREAD_GRAYSCALE)).astype("int64")
    img2 = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE)).astype("int64")
    print(MSE(img1, img2))

elif command == 'psnr':
    path1 = sys.argv[2]
    path2 = sys.argv[3]
    img1 = np.array(cv2.imread(path1, cv2.IMREAD_GRAYSCALE)).astype("int64")
    img2 = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE)).astype("int64")
    print(PSNR(img1, img2, 255))

elif command == 'ssim':
    path1 = sys.argv[2]
    path2 = sys.argv[3]
    img1 = np.array(cv2.imread(path1, cv2.IMREAD_GRAYSCALE)).astype("int64")
    img2 = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE)).astype("int64")
    print(SSIM(img1, img2, 255))

elif command == 'median':
    rad = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    img = np.array(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)).astype("int64")
    result_img = median_filter(img, rad)

elif command == 'gauss':
    sigma_d = float(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    img = np.array(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)).astype("int64")
    result_img = gauss_filter(img, sigma_d)

elif command == 'bilateral':
    sigma_d = float(sys.argv[2])
    sigma_r = float(sys.argv[3])
    input_path = sys.argv[4]
    output_path = sys.argv[5]
    img = np.array(cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)).astype("int64")
    result_img = bilateral_filter(img, sigma_d, sigma_r)

elif command == 'compare':
    path1 = sys.argv[2]
    path2 = sys.argv[3]
    img1 = np.array(cv2.imread(path1, cv2.IMREAD_GRAYSCALE)).astype("int64")
    img2 = np.array(cv2.imread(path2, cv2.IMREAD_GRAYSCALE)).astype("int64")
    result = detect_shift_and_rotate(img1, img2)
    if result < 0.9:
        print(0)
    else:
        print(1)

else:
    raise Exception('Некорректная команда!')

if output_path:
    cv2.imwrite(output_path, result_img)