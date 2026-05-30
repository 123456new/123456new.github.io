import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import Counter
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cpu')

INPUT_SIZE = 24
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 500
LAYERS = 2
DROP_RATE = 0.5
TIME_STEP = 24

from nnet.blstm import blstm
model = blstm(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LAYERS, DROP_RATE).to(device)

model_path = 'model/SLR/blstm_output500_input24x24.pkl'
checkpoint = torch.load(model_path, weights_only=False, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

label_dict = {}
with open('data/SLR_dataset/dictionary.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            idx = int(parts[0])
            if idx < OUTPUT_SIZE:
                label_dict[idx] = parts[1]
index_to_label = label_dict

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

# MediaPipe BlazePose landmark indices matching Kinect training data
JOINT_IDS = [
    11,   # LEFT_SHOULDER     → Kinect Joint 4
    13,   # LEFT_ELBOW         → Kinect Joint 5
    15,   # LEFT_WRIST         → Kinect Joint 6
    19,   # LEFT_INDEX         → Kinect Joint 7 (HandLeft)
    12,   # RIGHT_SHOULDER     → Kinect Joint 8
    14,   # RIGHT_ELBOW        → Kinect Joint 9
    16,   # RIGHT_WRIST        → Kinect Joint 10
    20,   # RIGHT_INDEX        → Kinect Joint 11 (HandRight)
    17,   # LEFT_PINKY         → Kinect Joint 21 (HandTipLeft)
    21,   # LEFT_THUMB         → Kinect Joint 22 (ThumbLeft)
    18,   # RIGHT_PINKY        → Kinect Joint 23 (HandTipRight)
    22,   # RIGHT_THUMB        → Kinect Joint 24 (ThumbRight)
]

# bone connections for visualization
BONES = [
    (0, 1), (1, 2), (2, 3),         # left arm
    (4, 5), (5, 6), (6, 7),         # right arm
    (2, 8), (2, 9), (6, 10), (6, 11),  # wrist to fingers
]

def abs2rel(data, crop_size):
    data_x = data[0::2]
    data_y = data[1::2]
    x_min, x_max = np.min(data_x), np.max(data_x)
    y_min, y_max = np.min(data_y), np.max(data_y)
    if x_max - x_min < 1e-6 or y_max - y_min < 1e-6:
        return data
    data[0::2] = (data_x - x_min) / (x_max - x_min) * crop_size
    data[1::2] = (data_y - y_min) / (y_max - y_min) * crop_size
    return data

FONT_PATH = r'C:\Windows\Fonts\simhei.ttf'
font_cn = ImageFont.truetype(FONT_PATH, 36)

def put_chinese(frame, text, pos, color=(0, 0, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font_cn, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'pose_landmarker_lite.task')
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
sequence = []
predictions = []
frame_idx = 0

print("Sign Language Recognition Demo (MediaPipe). Press Q to quit.")

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, frame_idx)
        frame_idx += 1

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            h, w = frame.shape[:2]
            data = np.zeros(24, dtype=np.float32)
            for i, joint_id in enumerate(JOINT_IDS):
                lm = landmarks[joint_id]
                data[i * 2] = lm.x * w
                data[i * 2 + 1] = lm.y * h
            sequence.append(abs2rel(data, 256))

            pts = [(int(landmarks[j].x * w), int(landmarks[j].y * h))
                   for j in JOINT_IDS]
            for i, j in BONES:
                cv2.line(frame, pts[i], pts[j], (0, 255, 0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        if len(sequence) > TIME_STEP:
            sequence.pop(0)

        word = ""
        if len(sequence) == TIME_STEP:
            seq_array = np.array(sequence, dtype=np.float32)
            tensor_input = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor_input)
                pred = torch.argmax(out[0, -1, :]).item()
            predictions.append(pred)
            if len(predictions) > 10:
                predictions.pop(0)
            smoothed = Counter(predictions).most_common(1)[0][0]
            word = index_to_label.get(smoothed, "Unknown")

        frame = put_chinese(frame, f"Sign: {word}", (20, 50))
        cv2.imshow("Sign Language MediaPipe Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
