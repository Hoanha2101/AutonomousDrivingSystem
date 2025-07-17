import cv2
import time
import serial
import numpy as np
import pygame
import sys
import os
from TRT_BRAIN import BRAIN, LOGIC_STEERING
import TRT_BRAIN

# ---------------- Cấu hình nguồn ----------------
video_path = "videos/start.mp4"
use_webcam = False
use_serial = False
have_serial = False
WEBCAME = 1

cap = cv2.VideoCapture(WEBCAME if use_webcam else video_path)
serial_port = None

# ---------------- Khởi tạo Pygame ----------------
pygame.init()

ret, frame = cap.read()
if not ret:
    print("Không mở được video")
    sys.exit()

screen_width, screen_height = 1536, 800
camera_width, camera_height = 600, 950
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Auto Car - FPT Lab")
font = pygame.font.SysFont("Arial", 24)
big_font = pygame.font.SysFont("Arial", 36)

# ---------------- Logo FPT ----------------------
logo_fpt_path = os.path.join("image_set/logofptuniversity.png")
logo_fpt_surface = pygame.image.load(logo_fpt_path)
logo_fpt_surface = pygame.transform.scale(logo_fpt_surface, (150, 58))

# ---------------- Logo car ----------------------
logo_car_path = os.path.join("image_set/car.png")
logo_car_surface = pygame.image.load(logo_car_path)
logo_car_surface = pygame.transform.scale(logo_car_surface, (280,200))

# ---------------- Trạng thái nút ----------------
ai_enabled = False
drawn_enabled = False

button_rects = {
    "ai": pygame.Rect(1326, 10, 200, 60),
    "drawn": pygame.Rect(1326, 90, 200, 60),
    "video": pygame.Rect(1326, 170, 200, 60),
    "serial": pygame.Rect(1326, 250, 200, 60),
    "fps": pygame.Rect(1326, 330, 200, 60),
    "debug": pygame.Rect(1326, 410, 200, 60),
    "refresh": pygame.Rect(1026, 10, 200, 60),
    "exit": pygame.Rect(1326, 730, 200, 60)
}

# ---------------- Màu sắc ----------------
def draw_button(rect, text, bg_color, text_color=(255, 255, 255)):
    pygame.draw.rect(screen, bg_color, rect, border_radius=10)
    label = font.render(text, True, text_color)
    text_rect = label.get_rect(center=rect.center)
    screen.blit(label, text_rect)

# ---------------- Vòng lặp chính ----------------
frame_count = 0
skip_frames = 5
running = True
Count_Warm_Up = 0
status_text = "_"
show_fps = False  # Trạng thái bật/tắt FPS
show_debug = False
show_refresh = False

# --- Thêm biến kiểm soát số frame cua ---
steer_count = 0
last_steer = "S"  # "X" (trái), "Y" (phải), "S" (thẳng)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rects["ai"].collidepoint(event.pos):
                ai_enabled = not ai_enabled
            elif button_rects["drawn"].collidepoint(event.pos):
                drawn_enabled = not drawn_enabled
            elif button_rects["video"].collidepoint(event.pos):
                use_webcam = not use_webcam
                cap.release()
                cap = cv2.VideoCapture(WEBCAME if use_webcam else video_path)
            elif button_rects["serial"].collidepoint(event.pos):
                use_serial = not use_serial
                if use_serial:
                    print("Use Serial")
                    serial_port = serial.Serial("COM8", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
                elif serial_port:
                    serial_port.close()
            elif button_rects["fps"].collidepoint(event.pos):
                show_fps = not show_fps
            elif button_rects["debug"].collidepoint(event.pos):
                show_debug = not show_debug
            elif button_rects["refresh"].collidepoint(event.pos):
                show_refresh = not show_refresh
                TRT_BRAIN.prev_pt1, TRT_BRAIN.prev_pt2, TRT_BRAIN.prev_distance = None, None, None
                
            elif button_rects["exit"].collidepoint(event.pos):
                running = False

    start_time = time.time()
    ret, img0 = cap.read()
    if not ret:
        break

    if use_serial and Count_Warm_Up <= 10:
        cap.read()
        print("---Warm up---")
        Count_Warm_Up += 1

        # Hiển thị thông báo "WARMING UP..."
        screen.fill((192, 192, 192))
        draw_button(button_rects["exit"], "Exit", (220, 20, 60))
        warm_text = big_font.render("WARMING UP...", True, (255, 0, 0))
        screen.blit(warm_text, (camera_width + 30, 10))
        pygame.display.update()
        continue

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    screen.fill((192, 192, 192))
    
    if ai_enabled:
        img0, pt1, pt2, center_point = BRAIN(img0, DRAWN=drawn_enabled, SHOW_DEBUG = show_debug)
        PUSH, DIRECTION_PUSH, img0 = LOGIC_STEERING(img0, center_point, DRAWN=drawn_enabled)

        # --- Bắt đầu kiểm soát số frame cua ---
        steer_cmd = DIRECTION_PUSH[0]  # "X", "Y", "S" (lấy ký tự đầu)
        send_cmd = False
        extra_cmds = []  # Danh sách lệnh cần gửi thêm nếu cần trả lái
        if steer_cmd in ("X", "Y"):  # Nếu là lệnh cua
            if steer_cmd == last_steer:
                if steer_count < 10:
                    steer_count += 1
                    send_cmd = True
                else:
                    # Đã đủ 10 frame, giữ nguyên trạng thái, không gửi thêm lệnh
                    send_cmd = False
            else:
                # Đổi hướng cua, reset lại đếm
                steer_count = 1
                last_steer = steer_cmd
                send_cmd = True
        elif steer_cmd == "S":  # Nếu là lệnh thẳng
            if steer_count > 0 and last_steer in ("X", "Y"):
                # Gửi lệnh ngược lại đúng bằng steer_count lần
                reverse_cmd = "Y" if last_steer == "X" else "X"
                for _ in range(steer_count):
                    extra_cmds.append(f"{reverse_cmd}:000")
                # Sau đó gửi lệnh S
                extra_cmds.append("S:000")
                steer_count = 0
                last_steer = "S"
                send_cmd = False  # Không gửi lệnh S ở dưới nữa, đã gửi ở extra_cmds
            else:
                steer_count = 0
                last_steer = "S"
                send_cmd = True
        else:
            # Lệnh không xác định, vẫn gửi
            send_cmd = True

        # Gửi các lệnh trả lái nếu có
        if use_serial and PUSH and serial_port and extra_cmds:
            for cmd in extra_cmds:
                status_text = f"→ Serial: {cmd}"
                serial_port.write(cmd.encode())
        # Gửi lệnh bình thường nếu cần
        if use_serial and PUSH and serial_port and send_cmd:
            status_text = f"→ Serial: {DIRECTION_PUSH}"
            serial_port.write(DIRECTION_PUSH.encode())
        else:
            status_text = f"→ Logic: {DIRECTION_PUSH}"


    
    if show_fps:
        # Hiển thị FPS
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(img0, f"FPS: {round(fps, 2)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Hiển thị video
    img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (camera_height, camera_width))
    img_rgb = cv2.flip(img_rgb, 1)
    img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    surface = pygame.surfarray.make_surface(img_rgb)
    screen.blit(surface, (10, 10))

    status_surface = font.render(status_text, True, (0, 0, 0))
    screen.blit(status_surface, (1200, 535))  

    # Logo
    screen.blit(logo_fpt_surface,(80, screen_height - 100))

    # CAR
    screen.blit(logo_car_surface,(950, 435))
    
    draw_button(button_rects["refresh"], f"Refresh: ON" if show_refresh else "Refresh: OFF", (0, 128, 0) if show_debug else (0, 0, 100))
    if show_refresh:
        show_refresh = not show_refresh
    
    # Nút Debug
    draw_button(button_rects["debug"], f"Debug: ON" if show_debug else "Debug: OFF", (0, 128, 0) if show_debug else (100, 100, 100))
    
    # Nút FPS
    draw_button(button_rects["fps"], f"FPS: ON" if show_fps else "FPS: OFF", (0, 128, 0) if show_fps else (100, 100, 100))

    # Nút AI
    draw_button(button_rects["ai"], f"AI: {'ON' if ai_enabled else 'OFF'}", (0, 200, 0) if ai_enabled else (200, 0, 0))

    # Nút Vẽ
    draw_button(button_rects["drawn"], f"DRAWN: {'ON' if drawn_enabled else 'OFF'}", (0, 0, 200) if drawn_enabled else (100, 100, 100))

    # Nút Video/Webcam
    draw_button(button_rects["video"], f"{'Webcam' if use_webcam else 'Video'}", (255, 165, 0))

    # Nút Serial
    draw_button(button_rects["serial"], f"Serial: {'ON' if use_serial else 'OFF'}", (128, 0, 128) if use_serial else (169, 169, 169))

    # Nút Exit
    draw_button(button_rects["exit"], "Exit", (220, 20, 60))

    pygame.display.update()

# ---------------- Dọn dẹp ----------------
cap.release()
pygame.quit()
cv2.destroyAllWindows()
