import cv2
import time
from TRT_BRAIN import BRAIN, BRAIN_MINI, LOGIC_STEERING
import serial
import matplotlib.pyplot as plt
import numpy as np


cap = cv2.VideoCapture(1)
have_serial = True

# cap = cv2.VideoCapture("videos/gr2.mp4")
# have_serial = False

if have_serial:
    serial_port = serial.Serial(
        port="COM8",
        baudrate=9600,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )

frame_count = 0
skip_frames = 5

# Bỏ qua vài khung đầu
for _ in range(30):
    cap.read()

while True:
    start_time = time.time()
    ret, img0 = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % skip_frames != 0:
        continue

    img0, pt1, pt2, center_point = BRAIN(img0, DRAWN = True)
    
    
    (PUSH, DIRECTION_PUSH, img0) = LOGIC_STEERING(img0, center_point, DRAWN = True)
    
    if have_serial:
        if PUSH:
            print('--------- Pushed by serial_port --------')
            print(DIRECTION_PUSH)
            serial_port.write(DIRECTION_PUSH.encode())
    else:
        print('--------- Pushed by serial_port --------')
        print(DIRECTION_PUSH)
    # Hiển thị FPS lên hình
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time if elapsed_time > 0 else 0
    
    cv2.putText(img0, f"fps: {round(fps, 3)}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        
    cv2.imshow("Result", img0)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
