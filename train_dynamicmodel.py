import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. LOCK THE RANDOMNESS (Consistent Accuracy)
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
DATA_PATH = os.path.join('MP_Dynamic_Data')
actions = np.array(sorted(os.listdir(DATA_PATH)))

# --- UPGRADED DATA REQUIREMENTS FOR 30 WORDS ---
no_sequences = 30     # You must collect 30 videos per word now!
sequence_length = 15  # Still using the ultra-fast 1.0 second buffer

print(f"Discovered {len(actions)} Dynamic Words: {actions}")
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            except FileNotFoundError:
                print(f"CRITICAL ERROR: Missing data in {action} / Video {sequence}")
                exit() # Immediately stop if data is missing, rather than crashing later
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 5% for testing, 95% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# ==========================================
# 3. THE 30-WORD LSTM ARCHITECTURE
# ==========================================
model = Sequential()

# Layer 1: Massive 256-neuron net with Dropout
model.add(LSTM(256, return_sequences=True, activation='relu', input_shape=(15, 126))) 
model.add(Dropout(0.2)) # Prevents the AI from cheating/memorizing

# Layer 2: Deep temporal processing
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))

# Layer 3: Condensing the timeline
model.add(LSTM(128, return_sequences=False, activation='relu'))

# Dense classification layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Output Layer: Automatically scales to however many folders you have
model.add(Dense(actions.shape[0], activation='softmax')) 

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==========================================
# 4. TRAINING & SAVING (AUTO-STOPPING)
# ==========================================
print(f"\n--- Starting High-Capacity Training for {len(actions)} Words ---")

# With 30 words and Dropout, the AI learns a bit slower. 
# Increased patience to 50 so it doesn't give up too early.
early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=300, restore_best_weights=True)

model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=1000, # Set massive limit, EarlyStopping will kill it when it's perfect
    callbacks=[early_stopping]
)

model.save('isl_dynamic_126.h5')
print("\nSuccess! Pro-Level Dynamic AI Model saved as 'isl_dynamic_126.h5'")