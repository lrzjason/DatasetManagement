import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Define the directory where the images are stored
input_dir = 'F:/ImageSet/openxl2_realism_test'

# Function to count fingers in an image
def count_fingers(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        finger_count = 0
        for hand_landmarks in results.multi_hand_landmarks:
            # Logic to determine if each finger is up
            # Thumb
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                finger_count += 1
            # Fingers
            for finger_tip_id in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                  mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
                if hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_tip_id - 2].y:
                    finger_count += 1
        return finger_count
    else:
        return 0

# Iterate over each image in the input directory and count fingers
finger_counts = {}
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    finger_counts[image_name] = count_fingers(image_path)

# Print the finger counts for each image
for image_name, count in finger_counts.items():
    print(f"{image_name}: {count} fingers detected")

# Release resources
hands.close()
