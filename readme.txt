🚀 Installation & Setup
1. Clone the repository

Bash
git clone [https://https://github.com/Abhinav3121/ISL-RECOGINITION-SYSYTEM.git]
cd ISL-Recognition-System
2. Install required dependencies
It is recommended to use a virtual environment.

Bash
pip install opencv-python mediapipe tensorflow scikit-learn pywin32 numpy
Note: Depending on your hardware, you can also install tensorflow-gpu for faster training times.

📖 Usage Guide
Step 1: Collect Data
Run the data collection tool to build your custom dictionary. The system expects 30 videos (15 frames each) per word to ensure robust lighting and angle variation.

Bash
python collect_data_live.py
Type the words you want to record separated by commas (e.g., Hello, Thank You, Please). Follow the on-screen prompts.

Step 2: Train the Neural Networks
Train the dynamic LSTM model on your recorded words:

Bash
python train_dynamic.py
The script uses Early Stopping to automatically halt training when the validation accuracy peaks, saving the optimal isl_dynamic_126.h5 model.

Step 3: Run the Application
Launch the live detection system:

Bash
python main_app.py
⌨️ HUD Controls
While the main application is running, use these keyboard shortcuts:

[M] - Toggle between DYNAMIC and STATIC modes.

[SPACE] - Add a space to the current sentence.

[ENTER] - Confirm the current staged word and add it to the sentence.

[S] - Speak the entire constructed sentence out loud via TTS.

[BACKSPACE] - Delete the last character or word.

[C] - Clear the screen, sentence, and history.

[Q] - Quit the application safely.
