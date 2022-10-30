import sys
import cv2
import numpy as np

def rotate(img, angle, direction):
    result_img = img
    if direction not in ['cw', 'ccw']:
        raise Exception('Некорректно введено направление поворота изображения!')
    if angle % 90 != 0:
        raise Exception("Некорректное число градусов в функции rotate()!")
    if angle % 360 == 0:
        return result_img
    if direction == 'ccw':
        direction = 1
    else:
        direction = -1
    if angle < 0:
        angle = -angle
        direction = -direction
    angle %= 360
    if angle == 90:
        if direction == 1:
            result_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            result_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        result_img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        result_img = cv2.rotate(img, cv2.ROTATE_180)
        if direction == 1:
            result_img = cv2.rotate(result_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            result_img = cv2.rotate(result_img, cv2.ROTATE_90_CLOCKWISE)
    return result_img

def mirroring(img, action):
    result_img = img
    if action == 'h':
        result_img = cv2.flip(img, 0)
    elif action == 'v':
        result_img = cv2.flip(img, 1)
    elif action == 'd':
        result_img = img.T
    elif action == 'cd':
        img = cv2.flip(img, 1)
        img = cv2.flip(img, 0)
        result_img = img.T
    else:
        raise Exception('Некорректный параметр action в функции mirroring()!')
    return result_img
 
def autocontrast(img):
    max_pixel = np.max(img)
    min_pixel = np.min(img)
    if min_pixel == max_pixel:
        return img
    return  (255 / (max_pixel - min_pixel) ) * (img - min_pixel)

def extract(img, left_x, top_y, width, height):
    shape = img.shape
    gap = max(-left_x, -top_y, left_x + width - shape[1], top_y + height - shape[0])
    if gap <= 0:
        return img[top_y: top_y + height, left_x: left_x + width]
    gapped_img = np.zeros((2 * gap + shape[0], 2 * gap + shape[1]))
    gapped_img[gap:gap + shape[0], gap:gap + shape[1]] = img
    return gapped_img[gap + top_y:gap + top_y + height, gap + left_x:gap + left_x + width]

def fixinterlace(img):
    copy_img = np.copy(img)
    k = img.shape[0] // 2
    for i in range(k):
        img[2 * i], img[2 * i + 1] = np.copy(img[2 * i + 1]), np.copy(img[2 * i])
    if vertical_variance(copy_img) < vertical_variance(img):
        return copy_img
    else:
        return img

def vertical_variance(arr):
    k = img.shape[0]
    vec = np.zeros((1, img.shape[1]))
    for i in range(k - 1):
        vec += np.abs(arr[i + 1] - arr[i])
    return np.sum(vec)

path = sys.argv[-2]
output_path = sys.argv[-1]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
result_img = img
command = sys.argv[1]

if command == 'rotate':
    clockwork = sys.argv[2]
    angle = int(sys.argv[3])
    result_img = rotate(img, angle, clockwork)

elif command == 'mirror':
    action = sys.argv[2]
    result_img = mirroring(img, action)

elif command == 'autocontrast':
    result_img = autocontrast(img)
    
elif command == 'extract':
    left_x = int(sys.argv[2])
    top_y = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    result_img = extract(img, left_x, top_y, width, height)

elif command == 'fixinterlace':
    result_img = fixinterlace(img)
    
else:
    raise Exception('Некорректная команда!')

cv2.imwrite(output_path, result_img)
