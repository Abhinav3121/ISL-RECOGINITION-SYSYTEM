import cv2
import numpy as np
import mediapipe as mp
import os
import time  

# ==========================================
# 1. DYNAMIC CONFIGURATION
# ==========================================
print("=== ISL STATIC DATA COLLECTION TOOL ===")
print("Type the letters or static signs you want to record, separated by commas.")
print("Example: A, B, C, D")
user_input = input("\nEnter signs: ")

# Clean up input and capitalize for alphabet consistency
word_list = [word.strip().upper() for word in user_input.split(',')]
actions = np.array(word_list)

# Number of frames to capture per static sign
no_samples = 300 

DATA_PATH = os.path.join('MP_Static_Data') 

print(f"\nSetting up folders for: {word_list}")
for action in actions: 
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# ==========================================
# 2. NORMALIZED EXTRACTION (Matches App Exactly)
# ==========================================
def extract_hand_keypoints(results):
    lh, rh = np.zeros(63), np.zeros(63)
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            
            # THE FIX: Safely extract Left/Right classification
            handedness = "Right" # Default fallback
            if results.multi_handedness and len(results.multi_handedness) > idx:
                 handedness = results.multi_handedness[idx].classification[0].label
                 
            wrist_x = hand.landmark[0].x
            wrist_y = hand.landmark[0].y
            wrist_z = hand.landmark[0].z
            
            normalized_pts = []
            for lm in hand.landmark:
                normalized_pts.extend([
                    lm.x - wrist_x, 
                    lm.y - wrist_y, 
                    lm.z - wrist_z
                ])
                
            pts = np.array(normalized_pts)
            if handedness == 'Right': lh = pts 
            else: rh = pts
            
    return np.concatenate([lh, rh])

# ==========================================
# 3. LIVE STATIC COLLECTION LOOP
# ==========================================
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    for action in actions:
        
        action_path = os.path.join(DATA_PATH, action)
        existing_files = [int(f.split('.')[0]) for f in os.listdir(action_path) if f.endswith('.npy')]
        start_idx = max(existing_files) + 1 if existing_files else 0

        # --- WAITING SCREEN ---
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            cv2.putText(frame, f'Get ready for STATIC sign: {action}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f'Will collect {no_samples} frames.', (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, 'Press "R" to start recording...', (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('ISL Static Data Collector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('r'):
                break

        # ==========================================
        # 3-SECOND COUNTDOWN SCREEN
        # ==========================================
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

                cv2.putText(frame, f'Get ready for STATIC sign: {action}', (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                h, w, _ = frame.shape
                cv2.putText(frame, str(countdown), (w//2 - 50, h//2 + 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 140, 255), 12)
                
                cv2.imshow('ISL Static Data Collector', frame)
                cv2.waitKey(1)

        # --- RECORDING SCREEN ---
        frames_collected = 0
        
        while frames_collected < no_samples:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                keypoints = extract_hand_keypoints(results)
                npy_path = os.path.join(action_path, str(start_idx + frames_collected))
                np.save(npy_path, keypoints)
                frames_collected += 1

            # UI Feedback
            cv2.putText(frame, f'RECORDING: {action}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, 'Move your hand slightly to capture different angles!', (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            
            # Progress bar visualization
            fill_width = int((frames_collected / no_samples) * 400)
            cv2.rectangle(frame, (10, 80), (10 + fill_width, 110), (0, 255, 0), -1)
            cv2.putText(frame, f'{frames_collected}/{no_samples}', (15, 103), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            cv2.imshow('ISL Static Data Collector', frame)
            cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
print("\nStatic Data Collection Complete! Run your static CNN training script next.")