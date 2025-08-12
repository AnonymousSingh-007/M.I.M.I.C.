🧠 M.I.M.I.C: Machine Intelligence Mimicking Cursor
"Looks real. Isn’t."
Welcome to M.I.M.I.C, a cutting-edge, machine learning-driven cursor spoofer designed to emulate human mouse movements with uncanny precision. Built for researchers, security enthusiasts, and tech tinkerers, M.I.M.I.C captures real user input, trains a neural network to replicate it, and deploys hyper-realistic cursor movements to challenge anti-bot and detection systems. Think of it as a digital doppelgänger for your mouse—part Iron Man tech, part Batman stealth.

---

🚀 Project Overview
M.I.M.I.C (Machine Intelligence Mimicking Cursor) leverages a Multi-Layer Perceptron (MLP) trained on real human mouse data to simulate natural cursor behavior. Whether you're testing anti-spoofing systems, exploring ML-driven automation, or just hacking around, M.I.M.I.C delivers a sleek, hacker-ish experience with rich console visuals and a modular, extensible codebase.
Core Objectives:

🖱️ Capture: Record authentic human mouse movements (X, Y, timestamp).
🤖 Learn: Train a neural network to predict realistic motion patterns.
🎯 Spoof: Simulate human-like cursor movements in real-time.
🧪 Test: Challenge detection systems with lifelike input patterns.

---

🛠️ Tech Stack

Language: Python 3.13.5
Libraries:
PyAutoGUI: For real-time cursor control.
Pandas: Data handling and CSV storage.
PyTorch: Neural network training (MLP).
Rich: Slick console visuals and UX.
scikit-learn: Data preprocessing and utilities.
OpenCV: Visual tracking enhancements.
Pygame: Optional visualization and input handling.

---

📁 Project Structure
mimic-cursor-spoofer/
├── data/
│   └── mouse_movements.csv      # Stored mouse movement data (time, x, y)
├── mimic/
│   ├── __init__.py              # Package initialization
│   ├── spoofer.py               # Cursor spoofing logic
│   ├── model.py                 # Neural network training and prediction
│   └── visuals.py               # Rich console UI and visualizations
├── main.py                      # Entry point for the application
├── requirements.txt             # Dependencies
└── .gitignore                   # Git ignore rules

---
✅ Current Features



Feature
Status
Description



Mouse Movement Data Collection
✅
Tracks cursor X, Y coordinates with timestamps.


CSV Data Storage
✅
Saves movement data for reproducibility and analysis.


Neural Network Training
✅
Uses a PyTorch MLP to learn movement patterns.


Real-Time Cursor Spoofing
✅
Simulates human-like cursor motion with pyautogui.


Smart Spoofing Duration
✅
Limits spoofing to ≤75% of recorded data time for realism.


Rich Console Visuals
✅
Interactive CLI with vibrant, hacker-style UX using rich.

---

🧬 How It Works
1. Data Collection

Step: User selects a recording duration (e.g., 15 seconds).
Process: Tracks cursor movements (X, Y coordinates + timestamp).
Output: Saves data as mouse_movements.csv in the data/ folder.
Format: time, x, y



2. Model Training

Architecture: Multi-Layer Perceptron (MLP) in PyTorch.
Input: Timestamp.
Output: Predicted (X, Y) cursor position.
Training: Learns smooth, human-like movement patterns from collected data.

3. Spoofing / Simulation

Step: User specifies a spoof duration (max 75% of recorded time).
Process: The trained model generates realistic (X, Y) coordinates.
Execution: pyautogui.moveTo(x, y) drives the cursor in real-time, mimicking human behavior.

4. Visual Feedback

Console: Rich, terminal-based UI with live progress bars, stats, and cyberpunk-inspired aesthetics.
Optional: Pygame/OpenCV for path visualization (planned for Phase 2).

---

🛠️ Getting Started
Prerequisites

Python 3.13.5
Git
A passion for hacking and ML 🤖

Installation

Clone the Repository:
git clone https://github.com/yourusername/mimic-cursor-spoofer.git
cd mimic-cursor-spoofer


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Run the Application:
python main.py

---

Usage

Launch main.py to access the interactive CLI.
Choose from:
Record: Capture mouse movements for a set duration.
Train: Train the MLP model on collected data.
Spoof: Simulate cursor movements using the trained model.


Enjoy the hacker-ish console visuals and real-time feedback.

---

🦸‍♂️ Why M.I.M.I.C?
M.I.M.I.C is more than a tool—it's a playground for ML enthusiasts, security researchers, and automation hackers. It combines the precision of Tony Stark's JARVIS with the stealth of Batman's utility belt. Whether you're stress-testing anti-bot systems or exploring the limits of ML-driven automation, M.I.M.I.C is your partner in crime (ethically, of course).

---

🛡️ Ethical Use
M.I.M.I.C is designed for research, testing, and educational purposes. Do not use it to bypass security systems without explicit permission. Always respect privacy and legal boundaries.

---

📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## "Code like Tony Stark, sneak like Batman."Dive into M.I.M.I.C and start spoofing the unspoofable. 🚀
