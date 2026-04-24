import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import queue
import os
from collections import deque
import win32com.client
import pythoncom

try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("CRITICAL WARNING: TensorFlow not found in your environment.")

class TTSEngine:
    def __init__(self):
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        pythoncom.CoInitialize()
        engine = win32com.client.Dispatch("SAPI.SpVoice")
        engine.Rate = 1 
        while True:
            text = self.q.get()
            if text is None: break
            engine.Speak(text)
            self.q.task_done()

    def speak(self, text):
        self.q.put(text)

tts = TTSEngine()

DYNAMIC_ACTIONS = np.array(sorted(os.listdir('MP_Dynamic_Data'))) if os.path.exists('MP_Dynamic_Data') else np.array([])
STATIC_ACTIONS = np.array(sorted(os.listdir('MP_Static_Data'))) if os.path.exists('MP_Static_Data') else np.array([])

print("========================================")
print(" LOADING NEURAL NETWORKS...")
try:
    lstm_model = load_model('isl_dynamic_126.h5') if len(DYNAMIC_ACTIONS) > 0 else None
    if lstm_model: print(f" [SUCCESS] Dynamic Model Loaded ({len(DYNAMIC_ACTIONS)} words)")
    cnn_model = load_model('isl_static_126.h5') if len(STATIC_ACTIONS) > 0 else None
    if cnn_model: print(f" [SUCCESS] Static Model Loaded ({len(STATIC_ACTIONS)} letters)")
except Exception as e:
    print(f" [WARNING] Model Load Error: {e}")
    lstm_model, cnn_model = None, None
print("========================================\n")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def extract_hand_keypoints(results):
    lh, rh = np.zeros(63), np.zeros(63)
    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            handedness = "Right" 
            if hasattr(results, 'multi_handedness') and results.multi_handedness:
                 if len(results.multi_handedness) > idx:
                     handedness = results.multi_handedness[idx].classification[0].label
            wrist_x, wrist_y, wrist_z = hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z
            normalized_pts = []
            for lm in hand.landmark:
                normalized_pts.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            pts = np.array(normalized_pts)
            if handedness == 'Right': lh = pts 
            else: rh = pts
    return np.concatenate([lh, rh])

def calculate_motion(sequence):
    if len(sequence) < 2: return 0
    seq_array = np.array(sequence)
    deltas = np.diff(seq_array, axis=0)
    return np.mean(np.abs(deltas))

def draw_custom_hud(frame, mode, fps, prediction, conf, motion, seq_len, word, sentence, history, hands_visible, static_counter, is_signing):
    h, w, _ = frame.shape
    overlay = frame.copy()
    bg_dark, bg_panel = (35, 25, 25), (50, 40, 40)
    lime, hud_accent_blue = (0, 255, 150), (255, 140, 0)
    text_dim, text_bright = (120, 110, 110), (220, 220, 220)
    
    cv2.rectangle(overlay, (0, 0), (w, 60), bg_dark, -1)                  
    cv2.rectangle(overlay, (0, h-120), (w, h), bg_dark, -1)               
    cv2.rectangle(overlay, (10, 70), (250, 320), bg_panel, -1)            
    cv2.rectangle(overlay, (10, 330), (250, 480), bg_panel, -1)           
    cv2.rectangle(overlay, (w-260, 70), (w-10, 350), bg_panel, -1)        
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.putText(frame, "ISL", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, lime, 3)
    cv2.putText(frame, "DETECTION SYSTEM NATIVE", (85, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_dim, 1)
    
    center_x = w // 2
    cv2.rectangle(frame, (center_x-80, 15), (center_x, 45), bg_panel, -1)
    cv2.rectangle(frame, (center_x, 15), (center_x+80, 45), hud_accent_blue if mode == "DYNAMIC" else lime, -1) 
    cv2.putText(frame, "STATIC", (center_x-70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_dim if mode == "DYNAMIC" else bg_dark, 2)
    cv2.putText(frame, "DYNAMIC", (center_x+5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bg_dark if mode == "DYNAMIC" else text_dim, 2)
    
    cv2.putText(frame, str(fps), (w - 80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, hud_accent_blue, 3) 
    cv2.putText(frame, "FPS", (w - 40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_dim, 1)
    
    cv2.putText(frame, "PREDICTION", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_dim, 1)
    if hands_visible:
        pred_color = lime if prediction != "LOW CONFIDENCE" else hud_accent_blue
        font_scale = 0.6 if prediction == "LOW CONFIDENCE" else 1.5
        cv2.putText(frame, prediction, (110, 190), cv2.FONT_HERSHEY_SIMPLEX, font_scale, pred_color, 2 if prediction == "LOW CONFIDENCE" else 3)
        
        cv2.circle(frame, (125, 240), 40, lime, 3)
        cv2.putText(frame, f"{conf:.0f}%", (105, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lime, 2)
        if mode == "STATIC" and static_counter > 0:
            progress = min(1.0, static_counter / 5.0) 
            end_angle = int(progress * 360)
            cv2.ellipse(frame, (125, 240), (50, 50), -90, 0, end_angle, hud_accent_blue, 3) 
            cv2.putText(frame, f"{(progress*0.3):.1f}s / 0.3s", (95, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_accent_blue, 1) 
    else:
        cv2.putText(frame, "NO HAND", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_dim, 2)
        cv2.putText(frame, "DETECTED", (60, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_dim, 2)
        cv2.circle(frame, (125, 260), 30, text_dim, 1)
        
    cv2.putText(frame, "STATUS", (15, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_dim, 1)
    
    if mode == "DYNAMIC":
        if is_signing and seq_len < 15:
            status_text, status_col, text_col = "RECORDING...", hud_accent_blue, bg_dark 
        elif is_signing and seq_len == 15:
            status_text, status_col, text_col = "ANALYZING...", lime, bg_dark
        else:
            status_text, status_col, text_col = "LISTENING", bg_dark, text_bright
    else:
        status_text, status_col, text_col = "STATIC MODE", bg_dark, text_bright
        
    cv2.rectangle(frame, (60, 370), (250, 400), status_col, -1)
    cv2.putText(frame, status_text, (70, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_col, 2)
    
    if mode == "DYNAMIC":
        cv2.putText(frame, f"BUFFER: {seq_len}/15", (15, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_dim, 1)
        cv2.line(frame, (85, 417), (240, 417), bg_dark, 4)
        if seq_len > 0:
            fill_width = int((seq_len / 15.0) * 155)
            bar_color = lime if seq_len == 15 else hud_accent_blue 
            cv2.line(frame, (85, 417), (85 + fill_width, 417), bar_color, 4)

    cv2.putText(frame, "MOTION", (15, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.3, hud_accent_blue, 1) 
    cv2.line(frame, (15, 455), (240, 455), text_dim, 2)
    motion_width = min(int(motion * 3000), 225)
    if motion_width > 0:
        cv2.line(frame, (15, 455), (15 + motion_width, 455), lime if motion > 0.005 else hud_accent_blue, 4) 
    cv2.putText(frame, f"{motion:.4f}", (15, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_dim, 1)
    
    cv2.putText(frame, "HISTORY", (w-250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_dim, 1)
    for i, hist_item in enumerate(reversed(history[-5:])): 
        y_pos = 130 + (i * 40)
        cv2.circle(frame, (w-240, y_pos-5), 4, lime, -1)
        cv2.putText(frame, hist_item[0], (w-220, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_bright, 2)
        cv2.putText(frame, f"{hist_item[1]:.0f}%", (w-60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_dim, 1)

    cv2.putText(frame, "WORD (STAGING)", (15, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_dim, 1)
    cv2.line(frame, (15, h-80), (150, h-80), text_dim, 1)
    cv2.putText(frame, word if word else "---", (160, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, lime, 3)
    
    cv2.putText(frame, "SENTENCE", (15, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_dim, 1)
    cv2.line(frame, (15, h-30), (150, h-30), text_dim, 1)
    cv2.putText(frame, sentence, (160, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_bright, 2)
    
    legend = "[M] MODE   [ENTER] CONFIRM WORD   [S] SPEAK   [BACKSPACE] DELETE   [C] CLEAR"
    cv2.putText(frame, legend, (15, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, lime, 1)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    mode = "DYNAMIC"
    sequence = deque(maxlen=15) 
    motion_sequence = deque(maxlen=15) 
    
    history = [] 
    current_prediction, current_confidence, motion_val = "---", 0.0, 0.0
    current_word, sentence = "", ""
    is_signing, cooldown, static_counter, missing_frames = False, 0, 0, 0
    last_static_pred = ""
    pTime, frame_count = 0, 0

    tts.speak("System initialized.")

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            frame = cv2.flip(frame, 1) 
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(), mp_styles.get_default_hand_connections_style())
            
            keypoints = extract_hand_keypoints(results)
            hands_visible = bool(results.multi_hand_landmarks)
            raw_wrist = np.zeros(2)
            if hands_visible:
                raw_wrist = np.array([results.multi_hand_landmarks[0].landmark[0].x, results.multi_hand_landmarks[0].landmark[0].y])
            
            if cooldown > 0: cooldown -= 1
                
            if mode == "DYNAMIC" and cooldown == 0:
                if hands_visible:
                    is_signing = True
                    missing_frames = 0 
                    
                    if frame_count % 2 == 0:
                        sequence.append(keypoints)
                        motion_sequence.append(raw_wrist)
                        
                        if len(sequence) == 15 and lstm_model and len(DYNAMIC_ACTIONS) > 0:
                            res = lstm_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                            idx = np.argmax(res)
                            
                            # --- DIAGNOSTIC X-RAY: REAL-TIME SCAN ---
                            # Only prints if it's over 30% sure to prevent spamming your terminal
                            if res[idx] > 0.30:
                                print(f"[SCANNING] AI thinks this is: '{DYNAMIC_ACTIONS[idx]}' ({res[idx]*100:.1f}%)")
                            
                            if res[idx] > 0.75: 
                                print(f">>> [AUTO-TRIGGER] SUCCESS: '{DYNAMIC_ACTIONS[idx]}' locked in! <<<\n")
                                current_prediction = DYNAMIC_ACTIONS[idx]
                                current_confidence = res[idx] * 100
                                current_word = current_prediction
                                history.append((current_word, current_confidence))
                                tts.speak(current_word)
                                
                                sequence.clear()
                                motion_sequence.clear()
                                cooldown = 20 
                                is_signing = False
                                
                    motion_val = calculate_motion(motion_sequence)
                        
                elif not hands_visible and is_signing:
                    missing_frames += 1 
                    
                    if missing_frames > 10: 
                        if len(sequence) == 15: 
                            if lstm_model and len(DYNAMIC_ACTIONS) > 0:
                                res = lstm_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                                idx = np.argmax(res)
                                
                                # --- DIAGNOSTIC X-RAY: DROP HAND SCAN ---
                                print(f"[HAND DROPPED] Final AI decision: '{DYNAMIC_ACTIONS[idx]}' ({res[idx]*100:.1f}%)\n")
                                
                                if res[idx] > 0.35: 
                                    current_prediction = DYNAMIC_ACTIONS[idx]
                                    current_confidence = res[idx] * 100
                                    current_word = current_prediction
                                    history.append((current_word, current_confidence))
                                    tts.speak(current_word)
                                else:
                                    current_prediction = "LOW CONFIDENCE"
                                    current_confidence = res[idx] * 100
                                        
                        is_signing = False
                        sequence.clear() 
                        motion_sequence.clear()
                        cooldown = 15 
                        missing_frames = 0 
                    
                elif not hands_visible and not is_signing:
                    if len(sequence) > 0: 
                        sequence.clear()
                        motion_sequence.clear()

            elif mode == "STATIC" and cooldown == 0:
                if hands_visible and frame_count % 2 == 0:
                    if cnn_model and len(STATIC_ACTIONS) > 0:
                        res = cnn_model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
                        idx = np.argmax(res)
                        
                        if res[idx] > 0.85: 
                            pred = STATIC_ACTIONS[idx]
                            current_prediction = pred
                            current_confidence = res[idx] * 100
                            
                            if pred == last_static_pred: static_counter += 1
                            else:
                                static_counter = 0
                                last_static_pred = pred
                                
                            if static_counter >= 5: 
                                current_word += pred
                                history.append((pred, current_confidence))
                                tts.speak(pred)
                                static_counter = 0
                                cooldown = 10 

            if len(history) > 20: history = history[-20:]

            cTime = time.time()
            fps = int(1 / (cTime - pTime)) if cTime - pTime > 0 else 0
            pTime = cTime
            
            draw_custom_hud(frame, mode, fps, current_prediction, current_confidence, motion_val, len(sequence), current_word, sentence, history, hands_visible, static_counter, is_signing)
            cv2.imshow("ISL Detection System Native", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('m'): 
                mode = "STATIC" if mode == "DYNAMIC" else "DYNAMIC"
                sequence.clear()
                motion_sequence.clear()
                is_signing = False
                current_word = ""
                tts.speak(f"{mode} mode")
            elif key == 32: 
                current_word += " "
            elif key == 13: 
                if current_word.strip():
                    sentence += current_word.strip() + " "
                    current_word = ""
            elif key == ord('s'): 
                if sentence.strip():
                    tts.speak(sentence.strip())
            elif key == 8: 
                if len(current_word) > 0: current_word = current_word[:-1]
                elif len(sentence) > 0: sentence = sentence[:-1]
            elif key == ord('c'):
                sentence, current_word = "", ""
                history.clear()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()