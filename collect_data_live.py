import cv2
import numpy as np
import os
import mediapipe as mp
import time

# ==========================================
# 1. DYNAMIC CONFIGURATION
# ==========================================
print("=== ISL DYNAMIC DATA COLLECTION TOOL ===")
print("Type the words you want to record, separated by commas.")
print("Example: Hello, Thank You, Doctor, Mother")
user_input = input("\nEnter words: ")

word_list = [word.strip().title() for word in user_input.split(',')]
actions = np.array(word_list)

DATA_PATH = os.path.join('MP_Dynamic_Data')
# --- UPGRADED PARAMETERS FOR 30-WORD DICTIONARY ---
no_sequences = 30     # You are now collecting 30 examples per word
sequence_length = 15  # The 1.0-second ultra-fast frame buffer

print(f"\nSetting up folders for: {word_list}")
for action in actions: 
    for sequence in range(0, 14):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# ==========================================
# 2. NORMALIZED EXTRACTION
# ==========================================
def extract_hand_keypoints(results):
    lh, rh = np.zeros(63), np.zeros(63)
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            handedness = "Right" 
            if hasattr(results, 'multi_handedness') and results.multi_handedness:
                 if len(results.multi_handedness) > idx:
                     handedness = results.multi_handedness[idx].classification[0].label
                 
            wrist_x = hand.landmark[0].x
            wrist_y = hand.landmark[0].y
            wrist_z = hand.landmark[0].z
            
            normalized_pts = []
            for lm in hand.landmark:
                normalized_pts.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                
            pts = np.array(normalized_pts)
            if handedness == 'Right': lh = pts 
            else: rh = pts
            
    return np.concatenate([lh, rh])

# ==========================================
# 3. COLLECTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for action in actions:
        for sequence in range(0, 14):
            
            # --- WAITING SCREEN ---
            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                
                cv2.putText(frame, f'DYNAMIC SIGN: {action}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f'Video: {sequence + 1} / {no_sequences}', (10, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, 'Press "R" to start recording...', (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('ISL Dynamic Data Collector', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    break

            # --- 3-SECOND COUNTDOWN SCREEN ---
            for countdown in range(3, 0, -1):
                start_time = time.time()
                while time.time() - start_time < 1.0:
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(image_rgb)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    cv2.putText(frame, f'DYNAMIC SIGN: {action}', (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    h, w, _ = frame.shape
                    cv2.putText(frame, str(countdown), (w//2 - 50, h//2 + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 140, 255), 12)
                    
                    cv2.imshow('ISL Dynamic Data Collector', frame)
                    cv2.waitKey(1)

            # --- RECORDING SCREEN ---
            frame_count = 0
            frames_collected = 0
            
            while frames_collected < sequence_length:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Capture only every 2nd frame (to match live app speed)
                if frame_count % 2 == 0:
                    keypoints = extract_hand_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frames_collected))
                    np.save(npy_path, keypoints)
                    frames_collected += 1
                
                cv2.putText(frame, f'RECORDING: {action}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                fill_width = int((frames_collected / sequence_length) * 400)
                cv2.rectangle(frame, (10, 80), (10 + fill_width, 110), (0, 255, 0), -1)
                cv2.putText(frame, f'Frame: {frames_collected}/{sequence_length}', (15, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                cv2.imshow('ISL Dynamic Data Collector', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
cap.release()
cv2.destroyAllWindows()
print("\nDynamic Data Collection Complete!")