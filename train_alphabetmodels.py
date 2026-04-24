import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOCK THE RANDOMNESS (Consistent Accuracy)
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
DATA_PATH = os.path.join('MP_Static_Data')
actions = np.array(sorted(os.listdir(DATA_PATH)))

print(f"Discovered {len(actions)} Static Signs: {actions}")
label_map = {label:num for num, label in enumerate(actions)}

data, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    for file_name in os.listdir(action_path):
        if file_name.endswith('.npy'):
            res = np.load(os.path.join(action_path, file_name))
            data.append(res)
            labels.append(label_map[action])

X = np.array(data)
y = to_categorical(labels).astype(int)

# 10% for testing, 90% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# ==========================================
# 3. DENSE NEURAL NETWORK ARCHITECTURE
# ==========================================
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(126,)))
model.add(Dropout(0.2)) # Brain-damage layer to prevent memorizing
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==========================================
# 4. TRAINING & SAVING
# ==========================================
print("\n--- Starting Consistent Static Network Training ---")
model.fit(X_train, y_train, epochs=100)

model.save('isl_static_126.h5')
print("\nSuccess! Static AI Model saved as 'isl_static_126.h5'")