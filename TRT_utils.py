import numpy as np
import cv2


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    #print(sem_img.shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
     
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)

def driving_area_mask_trt(seg):
    # Chọn class có xác suất cao nhất theo trục class (dim=1), kết quả: (1, 384, 640)
    seg_mask = np.argmax(seg, axis=1)[0]  # (384, 640)

    # Resize về cùng kích thước ảnh gốc (720, 1280)
    seg_mask_resized = cv2.resize(seg_mask.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST)
    return seg_mask_resized  # (720, 1280)

def lane_line_mask_trt(ll):
    # ll shape: (1, 1, 384, 640)
    ll_mask = ll[0, 0]  # (384, 640)

    # Resize về (720, 1280)
    ll_mask_resized = cv2.resize(ll_mask.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST)
    return ll_mask_resized  # (720, 1280)

def crosses_vertical_line(p1, p2, x_line, buffer=0):
    return min(p1[0], p2[0]) <= x_line + buffer and max(p1[0], p2[0]) >= x_line - buffer

def draw_masks_on_black(img_shape, da_seg_mask):
    black_image = np.zeros(img_shape, dtype=np.uint8)
    da_mask = da_seg_mask > 0
    black_image[da_mask] = (0, 0, 255)
    return black_image
