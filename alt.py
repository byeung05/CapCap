import cv2
import numpy as np
import time
import requests
import os
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Find the absolute path to the haarcascade file
script_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(script_dir, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

try:
    sound = pygame.mixer.Sound("capcap.wav")
except pygame.error as e:
    print(f"Error loading sound: {e}. Disabling audio.")
    sound = None

alternate_img = cv2.imread('angryCapy.png')
if alternate_img is None:
    print("Error loading alternate image. Exiting.")
    exit()

capybara_img = cv2.imread('happyCapy.PNG', cv2.IMREAD_UNCHANGED)
if capybara_img is None:
    print("Error loading capybara image. Exiting.")
    exit()

# Initialize counters and tracking variables
left_away_count = 0
right_away_count = 0
look_away_total = 0
last_direction = "forward"
last_look_away_time = time.time()
cooldown_period = 1
audio_played = False

capybara_x = 2
capybara_y = 2
capybara_width = 500
capybara_height = 500
capybara_img = cv2.resize(capybara_img, (capybara_width, capybara_height))

def overlay_image_alpha(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    h, w = overlay.shape[0], overlay.shape[1]

    if x >= background_width or y >= background_height:
        return background

    if x + w > background_width:
        w = background_width - x
    if y + h > background_height:
        h = background_height - y

    overlay_roi = overlay[0:h, 0:w]
    background_roi = background[y:y+h, x:x+w]

    if overlay_roi.shape[2] == 4:
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        background_roi = background_roi * (1 - alpha) + overlay_roi[:, :, :3] * alpha

    background[y:y+h, x:x+w] = background_roi
    return background

def increment_lookaway():
    try:
        response = requests.post("http://127.0.0.1:5000/update_lookaway")
        if response.status_code == 200:
            print("Look-away count updated.")
        else:
            print(f"Failed to update look-away count. Status code: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f"Error connecting to Flask app: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during increment_lookaway: {e}")

def play_sound():
    global sound
    if sound:
        try:
            sound.play()
        except pygame.error as e:
            print(f"Error playing sound: {e}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open any camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    overlay = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if (look_away_total % 8) < 4:
        image_to_display = capybara_img
    else:
        image_to_display = alternate_img

    if look_away_total % 4 == 0 and look_away_total > 0 and not audio_played:
        play_sound()
        audio_played = True
    elif look_away_total % 4 != 0:
        audio_played = False

    frame = overlay_image_alpha(frame, image_to_display, capybara_x, capybara_y)

    for (x, y, w, h) in faces:
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)

        center_x = x + w // 2
        frame_center_x = frame.shape[1] // 2

        current_time = time.time()

        if center_x > frame_center_x + w//4 and last_direction != "right" and (current_time - last_look_away_time) > cooldown_period:
            right_away_count += 1
            increment_lookaway()
            look_away_total += 1
            last_direction = "right"
            last_look_away_time = current_time
        elif center_x < frame_center_x - w//4 and last_direction != "left" and (current_time - last_look_away_time) > cooldown_period:
            left_away_count += 1
            increment_lookaway()
            look_away_total += 1
            last_direction = "left"
            last_look_away_time = current_time
        elif abs(center_x - frame_center_x) <= w//4:
            last_direction = "forward"

        if last_direction == "right":
            cv2.putText(frame, "Looking Right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif last_direction == "left":
            cv2.putText(frame, "Looking Left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Looking Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    alpha = 0.4
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, f"Left Count: {left_away_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Right Count: {right_away_count}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Total Look Aways: {look_away_total}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Head Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()