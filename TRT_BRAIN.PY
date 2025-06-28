import cv2
import numpy as np
import torch
import pycuda.driver as cuda
import pycuda.autoinit
from TensorrtBase_init import TensorrtBase
import matplotlib.pyplot as plt
import time
from TRT_utils import letterbox, driving_area_mask_trt, lane_line_mask_trt, crosses_vertical_line, draw_masks_on_black
import math

input_name = ['input']  # <- Đúng định dạng
output_names = ['pred0', 'seg' , 'pred1', 'll','pred2']

net = TensorrtBase("data/weights/model_384x640_FP16.trt",
                   input_names=input_name,
                   output_names=output_names,
                   max_batch_size=1)

# Tạo input mẫu
input_sample = torch.randn(1, 3, 384, 640).numpy().astype(np.float16)

net.context.set_input_shape("input", (1, 3, 384, 640))


if 'prev_distance' not in globals():
    prev_distance = None
    prev_pt1 = prev_pt2 = None

width, height = (1280, 720)
car_length_padding = 100

car_point_left  = (car_length_padding, height)
car_point_right = (width - car_length_padding, height)
car_center_bottom = ((car_point_left[0] + car_point_right[0]) // 2, height)
car_center_top = (car_center_bottom[0], 0)

frame_count = 0
skip_frames = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

rightmost_line_x_prev = None
xs_prev, ys_prev = None, None

MOOC = None
# Hàm tính khoảng cách từ tâm bounding box đến MOOC
# Hàm tính khoảng cách từ tâm bounding box đến MOOC
def distance_to_mooc(contour, mooc):
    x, y, w, h = cv2.boundingRect(contour)
    cx, cy = x + w // 2, y + h // 2
    return np.linalg.norm(np.array([cy, cx]) - np.array(mooc))


def BRAIN(img0, DRAWN = False, SHOW_DEBUG = False):
    global prev_distance, prev_pt1, prev_pt2, MOOC, rightmost_line_x_prev, final_mask, xs_prev, ys_prev
    
    img0 = cv2.resize(img0, (1280, 720))
    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img, _, _ = letterbox(img0, 640, stride=32)
    img = img[:, :, ::-1].copy().transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).cpu().numpy()

    net.cuda_ctx.push()
    pre0, seg, pre1, ll, pre2 = net.infer(img_tensor)
    net.cuda_ctx.pop()

    da_seg_mask = driving_area_mask_trt(seg) 
    ll_seg_mask = lane_line_mask_trt(ll) 
    

    black_with_masks = draw_masks_on_black(img0.shape, da_seg_mask)
    line_with_masks = draw_masks_on_black(img0.shape, ll_seg_mask)       
    
    # Cắt vùng quan trọng
    h = black_with_masks.shape[0]
    crop_y_start, crop_y_end = h // 2 + 100, h - 200
    y_offset = crop_y_start
    cropped_da = black_with_masks[crop_y_start:crop_y_end]
    cropped_line = line_with_masks[crop_y_start:crop_y_end]
    # cropped_line = cv2.cvtColor(cropped_line, cv2.COLOR_BGR2RGB)
    if SHOW_DEBUG:
        cv2.imshow("Debug Screen - da", cropped_da)
        cv2.imshow("Debug Screen - ll", cropped_line)
        

    # Tìm đường line đỏ (lane)
    red_mask = cv2.inRange(cropped_line, (0, 0, 200), (50, 50, 255))
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if not contours:
    #     print()
    # rightmost_line_x = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
    
    if contours:
        rightmost_line_x = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours])
        rightmost_line_x_prev = rightmost_line_x  # Cập nhật giá trị mới
    else:
        if rightmost_line_x_prev is not None:
            rightmost_line_x = rightmost_line_x_prev  # Dùng giá trị trước đó
        else:
            rightmost_line_x = 0  # fallback mặc định nếu chưa có gì

    # Tìm vùng driving đỏ nằm bên trái đường line
    da_mask = cv2.inRange(cropped_da, (0, 0, 200), (50, 50, 255))
    da_contours, _ = cv2.findContours(da_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [cnt for cnt in da_contours if cv2.boundingRect(cnt)[0] < rightmost_line_x]
    final_mask = np.zeros_like(cropped_da)
    if filtered:
        if MOOC is None:
            h_final_mask, w_final_mask = da_mask.shape
            MOOC = (h_final_mask, w_final_mask // 2)
        # Chọn contour gần MOOC nhất
        best_contour = min(filtered, key=lambda c: distance_to_mooc(c, MOOC))

        mask_clean = np.zeros_like(da_mask)
        cv2.drawContours(mask_clean, [best_contour], -1, 255, cv2.FILLED)
        final_mask = cv2.bitwise_and(cropped_da, cropped_da, mask=mask_clean)
    
    # cv2.imshow("mm", final_mask)
    
    # Lấy điểm cao nhất có màu đỏ
    ys, xs = np.where(final_mask[:, :, 2] == 255)
    # if xs.size == 0:
    #     print()

    if len(xs) == 0 or len(ys) == 0:
        if xs_prev is not None and ys_prev is not None:
            xs, ys = xs_prev, ys_prev
    else:
    # Cập nhật giá trị mới
        xs_prev, ys_prev = xs, ys

    min_y = np.min(ys)
    top_points = [(x, y) for x, y in zip(xs, ys) if y == min_y]
    leftmost = min(top_points, key=lambda p: p[0])
    rightmost = max(top_points, key=lambda p: p[0])

    pt1 = (leftmost[0], leftmost[1] + y_offset)
    pt2 = (rightmost[0], rightmost[1] + y_offset)
    current_distance = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5

    height, width = img0.shape[:2]
    center_x = width // 2
    valid = True

    if  prev_distance is not None:
        diff_ratio = abs(current_distance - prev_distance) / prev_distance
        if diff_ratio > 0.5:
            valid = False
    if not crosses_vertical_line(pt1, pt2, center_x, buffer=150):
        valid = False

    if not valid and prev_pt1 is not None:
        pt1, pt2 = prev_pt1, prev_pt2
        current_distance = prev_distance

    prev_pt1, prev_pt2, prev_distance = pt1, pt2, current_distance
    
    center_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

    if DRAWN:
        img0[da_seg_mask > 0] = (0, 255, 0)
        img0[ll_seg_mask > 0] = (35, 0, 200)
        
        # cv2.circle(img0, car_point_left, 10, (50, 100, 255), -1)
        # cv2.circle(img0, car_center_bottom, 10, (50, 100, 255), -1)
        # cv2.circle(img0, car_point_right, 10, (50, 100, 255), -1)
        # cv2.circle(img0, (car_center_top[0], center_point[1]), 10, (50, 100, 255), -1)

        # cv2.line(img0, car_center_bottom, center_point, (200, 0, 100), 10)
        # cv2.line(img0, car_center_bottom, (car_center_top[0], center_point[1]), (20, 0, 200), 10)
        
            
        # cv2.circle(img0, pt1, 10, (0, 0, 0), -1)
        # cv2.circle(img0, pt2, 10, (0, 0, 0), -1)
        
        # Độ dài hướng mũi tên
        arrow_length = 30

        # Hướng mũi tên cho pt1 (ví dụ: hướng xuống)
        pt1_end = (pt1[0], pt1[1] - arrow_length)
        cv2.arrowedLine(img0, pt1, pt1_end, (0, 0, 255), thickness=4, tipLength=0.4)

        # Hướng mũi tên cho pt2 (ví dụ: hướng lên)
        pt2_end = (pt2[0], pt2[1] - arrow_length)
        cv2.arrowedLine(img0, pt2, pt2_end, (255, 0, 0), thickness=4, tipLength=0.4)

        
        #---------------
        
        # cv2.line(img0, pt1, pt2, (0, 0, 0), 2)
        # cv2.line(img0, center_point, (car_center_top[0], center_point[1]), (0, 0, 255), 2)
        # cv2.circle(img0, center_point, 10, (255, 0, 0), -1)

        # cv2.putText(img0, f"Left: {leftmost}", (pt1[0], pt1[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        # cv2.putText(img0, f"Right: {rightmost}", (pt2[0], pt2[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        
    return img0, pt1, pt2, center_point



DIRECTION_LEFT = "X"
DIRECTION_RIGHT = "Y"
DIRECTION_STRAIGHT = "S"

direction_control = "S"
angle_deg_control = 0

REDUCE_LIST = []

def draw_arrow_icon(img, direction, position, color=(0, 0, 0)):
    x, y = position
    length = 60  

    if direction == DIRECTION_LEFT:
        start_point = (x + length, y)
        end_point = (x, y)
    elif direction == DIRECTION_RIGHT:
        start_point = (x, y)
        end_point = (x + length, y)
    elif direction == DIRECTION_STRAIGHT:
        start_point = (x, y + length)
        end_point = (x, y)
    else:
        return  # Không xác định

    cv2.arrowedLine(img, start_point, end_point, color, thickness=10, tipLength=0.5)


def DECIDE_STEERING(direction, angle_deg):
    global direction_control, angle_deg_control

    PUSH = False
    DIRECTION_PUSH = "S"

    if direction != DIRECTION_STRAIGHT:
        if direction != direction_control:
            PUSH = True
            DIRECTION_PUSH = f"{direction}:000"
            direction_control = direction
            angle_deg_control = angle_deg

        elif direction == direction_control and int(angle_deg) > int(angle_deg_control):
            PUSH = True
            DIRECTION_PUSH = f"{direction}:000"
            direction_control = direction
            angle_deg_control = angle_deg
            
        elif direction == direction_control and int(angle_deg) <= int(angle_deg_control):
            PUSH = True
            DIRECTION_PUSH = f"x:000"
            direction_control = direction
            angle_deg_control = angle_deg

    else:
        PUSH = False
        DIRECTION_PUSH = "S"
        direction_control = direction
        angle_deg_control = angle_deg

    return PUSH, DIRECTION_PUSH

def DECIDE_STEERING_PRO(direction, angle_deg):
    global direction_control, angle_deg_control

    PUSH = False
    DIRECTION_PUSH = "S"

    if direction != DIRECTION_STRAIGHT:
        if direction != direction_control:
            PUSH = True
            DIRECTION_PUSH = f"{direction}:000"
            direction_control = direction
            angle_deg_control = angle_deg

        elif direction == direction_control and int(angle_deg) > int(angle_deg_control):
            PUSH = True
            DIRECTION_PUSH = f"{direction}:000"
            direction_control = direction
            angle_deg_control = angle_deg
            
        elif direction == direction_control and int(angle_deg) <= int(angle_deg_control):
            PUSH = True
            DIRECTION_PUSH = f"x:000"
            direction_control = direction
            angle_deg_control = angle_deg

    else:
        PUSH = False
        DIRECTION_PUSH = "S"
        direction_control = direction
        angle_deg_control = angle_deg

    return PUSH, DIRECTION_PUSH
        
def LOGIC_STEERING(img0, center_point, DRAWN=False):
    car_center_bottom = (img0.shape[1] // 2, img0.shape[0] - 1)

    dx = center_point[0] - car_center_bottom[0]
    dy = car_center_bottom[1] - center_point[1]
    angle_rad = math.atan2(dx, dy)
    angle_deg = angle_rad * 180 / math.pi

    # Xác định hướng và màu
    threshold = 5
    if angle_deg < -threshold:
        direction = DIRECTION_LEFT
        arc_color = (0, 0, 255)  # đỏ
    
    elif angle_deg > threshold:
        direction = DIRECTION_RIGHT
        arc_color = (255, 0, 0)  # xanh
        
    else:
        direction = DIRECTION_STRAIGHT
        arc_color = (0, 0, 0)    # đen
            
    PUSH, DIRECTION_PUSH = DECIDE_STEERING(direction, abs(angle_deg))
    
    # 4. Vẽ icon điều hướng
    draw_arrow_icon(img0, direction, (car_center_top[0], car_center_top[1] + 250), arc_color)
    
    # if DRAWN:
        # # 1. Vẽ điểm và hướng
        # cv2.circle(img0, car_center_bottom, 5, (255, 0, 0), -1)
        # cv2.circle(img0, center_point, 5, (0, 255, 0), -1)
        # cv2.line(img0, car_center_bottom, center_point, (0, 0, 255), 2)

        # # 2. Vẽ cung góc lái
        # radius = 50
        # base_angle = -90
        # end_angle = base_angle + int(angle_deg)
        # cv2.ellipse(img0, car_center_bottom, (radius, radius), 0, base_angle, end_angle, arc_color, 2)

        # # 3. Ghi góc lái
        # text = f"{angle_deg:.1f} deg"
        # text_pos = (car_center_bottom[0] + 10, car_center_bottom[1] - radius - 10)
        # cv2.putText(img0, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, arc_color, 2)


        # # 5. Vẽ mũi tên định hướng (mô phỏng hướng xe)
        # arrow_tip = (car_center_top[0], center_point[1])
        # arrow_start = (car_center_bottom[0], car_center_bottom[1])
        # cv2.arrowedLine(img0, arrow_start, arrow_tip, (0, 255, 255), 3, tipLength=0.3)

        # 6. Hiển thị lệnh điều khiển
        # if PUSH:
        #     cv2.putText(img0, DIRECTION_PUSH, (img0.shape[1]//2 - 100, img0.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    return PUSH, DIRECTION_PUSH, img0


def BRAIN_MINI(img0, DRAWN=False):
    global prev_distance, prev_pt1, prev_pt2, MOOC

    img0 = cv2.resize(img0, (1280, 720))

    # Preprocess
    img, _, _ = letterbox(img0, 640, stride=32)
    img = img[:, :, ::-1].copy().transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).float().div(255.0).unsqueeze(0).cpu().numpy()

    # Model inference
    net.cuda_ctx.push()
    _, seg, _, _, _ = net.infer(img_tensor)
    net.cuda_ctx.pop()

    da_seg_mask = driving_area_mask_trt(seg)  # Binary mask [0,1]
    h = da_seg_mask.shape[0]
    crop_y_start, crop_y_end = h // 2 + 100, h - 200
    y_offset = crop_y_start
    cropped_mask = da_seg_mask[crop_y_start:crop_y_end]

    # Convert to uint8 for contour detection
    mask_uint8 = (cropped_mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img0, None, None, None

    if MOOC is None:
        h_final, w_final = mask_uint8.shape
        MOOC = (h_final, w_final // 2)

    best_contour = min(contours, key=lambda c: distance_to_mooc(c, MOOC))

    # Create clean mask from best contour
    mask_clean = np.zeros_like(mask_uint8)
    cv2.drawContours(mask_clean, [best_contour], -1, 255, cv2.FILLED)
    

    # Hiển thị kết quả vùng phân đoạn
    # cv2.imshow("Vung phan doan tot nhat", mask_clean)

    ys, xs = np.where(mask_clean == 255)
    if xs.size == 0:
        return img0, None, None, None

    min_y = np.min(ys)
    top_points = [(x, y) for x, y in zip(xs, ys) if y == min_y]
    leftmost = min(top_points, key=lambda p: p[0])
    rightmost = max(top_points, key=lambda p: p[0])

    pt1 = (leftmost[0], leftmost[1] + y_offset)
    pt2 = (rightmost[0], rightmost[1] + y_offset)
    current_distance = np.linalg.norm(np.array(pt1) - np.array(pt2))

    height, width = img0.shape[:2]
    center_x = width // 2
    valid = True

    if prev_distance is not None:
        diff_ratio = abs(current_distance - prev_distance) / prev_distance
        if diff_ratio > 0.5:
            valid = False

    if not crosses_vertical_line(pt1, pt2, center_x, buffer=150):
        valid = False

    if not valid and prev_pt1 is not None:
        pt1, pt2 = prev_pt1, prev_pt2
        current_distance = prev_distance

    prev_pt1, prev_pt2, prev_distance = pt1, pt2, current_distance
    center_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)

    if DRAWN:
        img0[da_seg_mask > 0] = (230, 230, 230)
        cv2.circle(img0, pt1, 10, (0, 0, 0), -1)
        cv2.circle(img0, pt2, 10, (0, 0, 0), -1)
        cv2.line(img0, pt1, pt2, (0, 0, 0), 2)
        cv2.circle(img0, center_point, 10, (255, 0, 0), -1)
        cv2.putText(img0, f"Left: {leftmost}", (pt1[0], pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(img0, f"Right: {rightmost}", (pt2[0], pt2[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img0, pt1, pt2, center_point


