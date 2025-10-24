
# 🧠 M.I.M.I.C — Machine Intelligence Mimicking Cursor

> **“Looks real. Isn’t.”**

Welcome to **M.I.M.I.C.**, a machine-learning-driven cursor spoofer that emulates human mouse movements with high realism. Built for researchers, security testers, and tinkerers, M.I.M.I.C captures real user input, trains a sequence model to replicate it, and can deploy hyper-realistic cursor movements for evaluation and experimentation.

> **Note:** This project is intended for research, testing, and educational purposes only. Do **not** use it to bypass security systems or on systems you don't have explicit permission to test. See the **Ethical Use** section below.

---

## 🚀 Project Overview

**M.I.M.I.C.** (Machine Intelligence Mimicking Cursor) uses sequence modeling (LSTM-based seq2seq) trained on recorded mouse movements to generate realistic cursor trajectories.

**Core objectives**

* 🖱️ **Capture** — record authentic human mouse movements (normalized deltas, speed, dt, curvature).
* 🤖 **Learn** — train a sequence-to-sequence model to predict motion patterns.
* 🎯 **Spoof** — generate and replay human-like cursor motion in real time.
* 🧪 **Test** — provide realistic input for anti-spoofing/behavioral analysis.

---

## 🛠️ Tech stack

* Language: **Python 3.x**
* Libraries:

  * `pyautogui` — real-time cursor control
  * `pandas` — data handling and CSV IO
  * `numpy` — numerical ops
  * `torch` (PyTorch) — model training & inference
  * `scikit-learn` — scaling/preprocessing
  * `rich` — console UI and visual feedback
  * `matplotlib` — plotting trajectories
  * `joblib` — save/load scalers
  * `scipy` (optional) — smoothing (Gaussian filter)

> Optional: `pygame`/`opencv` for advanced visualization (not required).

---

## 📁 Recommended project structure

```
mimic-cursor-spoofer/
├── data/
│   └── mouse_data.csv         # Recorded, preprocessed movement data
├── mimic/
│   ├── __init__.py
│   ├── collector.py           # Data collection
│   ├── model.py               # Seq2Seq model
│   ├── lstm_trainer.py        # Training loop + utils
│   ├── spoofer.py             # Generation + live cursor movement
│   ├── evaluator.py           # Plotting / evaluation utils
│   └── visuals.py             # Rich UI helpers
├── main.py                    # CLI entry point
├── config.json                # Optional config/ hyperparameters
├── requirements.txt
└── README.md
```

---

## ✅ Current features

|                      Feature | Status | Description                                                          |
| ---------------------------: | :----: | :------------------------------------------------------------------- |
|  Mouse movement data capture |    ✅   | Records normalized deltas, dt, speed, acceleration, curvature.       |
|             CSV data storage |    ✅   | Saves preprocessed movement samples for reproducibility.             |
| Neural sequence model (LSTM) |    ✅   | Seq2Seq LSTM-based model to learn temporal patterns.                 |
|    Real-time cursor spoofing |    ✅   | Replays generated cursor movement via `pyautogui`.                   |
|        Smart spoofing bounds |    ✅   | Clamps deltas/dt and respects screen bounds to avoid runaway cursor. |
|         Rich console visuals |    ✅   | Terminal UI, progress bars, success/error banners using `rich`.      |

---

## 🧬 How it works (high level)

### 1) Data collection

* The collector records mouse movements at a configured sample rate (e.g., 100 Hz).
* Output features typically include: `dx_norm`, `dy_norm`, `speed`, `dt`, `curvature`, `x_accel_norm`, `y_accel_norm`.
* Data is saved to `data/mouse_data.csv`.

### 2) Model training

* A Seq2Seq LSTM consumes sequences of length `seq_len` and predicts `pred_horizon` timesteps ahead.
* Input = several features per timestep. Output = predicted future feature vectors.
* Training uses a scaler (StandardScaler) saved to disk and includes validation/early stopping.

### 3) Spoofing / simulation

* The trained model is seeded with the latest `seq_len` recorded samples and generates deltas autoregressively.
* Generated normalized deltas are denormalized and applied to the cursor with `pyautogui.moveTo` (or `moveRel`) at safe dt intervals.
* Movements are clamped to screen bounds; dt and step sizes are clipped to realistic ranges.

### 4) Evaluation

* The evaluator plots recorded vs generated trajectories (positions derived from cumulative deltas) and saves an image for inspection.

---

## 🛠️ Getting started

### Prerequisites

* Python 3.8+ (the code is compatible with modern Python 3.x; if you use `3.13.5` ensure your environment matches available packages)
* pip, git

### Installation

```bash
git clone https://github.com/yourusername/mimic-cursor-spoofer.git
cd mimic-cursor-spoofer

python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

### Run the app

```bash
python main.py
```

The interactive CLI will guide you to record data, train the model, and run spoofing.

---

## Usage summary

* **Record**: choose a recording duration to capture mouse data into `data/mouse_data.csv`.
* **Train**: train a Seq2Seq model (saves model to `models/mimic_model.pt`, scaler to `models/mimic_scaler.pkl`).
* **Spoof**: load model + scaler, generate movements, and replay them live.
* **Evaluate**: plotted trajectories saved to `models/trajectories.png`.

---

## 🔧 Configuration & tuning recommendations

Add a `config.json` or edit the defaults in `main.py` for reproducible runs. Example (recommended starting point):

```json
{
  "model_type": "seq2seq",
  "epochs": 120,
  "num_layers": 2,
  "hidden_size": 128,
  "pred_horizon": 15,
  "seq_len": 150,
  "lr": 0.0008,
  "dropout": 0.3,
  "batch_size": 64,
  "teacher_forcing_ratio": 0.5
}
```

### Parameters to tune (suggested ranges)

* `seq_len`: **100–200** — longer sequence = more context (memory cost ↑).
* `pred_horizon`: **10–30** — horizon to predict at each step; shorter is more stable.
* `hidden_size`: **128–256** — model capacity.
* `num_layers`: **1–3** — depth of LSTM.
* `dropout`: **0.2–0.5** — regularization.
* `batch_size`: **32–128** — GPU/CPU memory dependent.
* `lr`: **5e-4 – 1e-3** — learning rate.
* `teacher_forcing_ratio`: **0.3–0.7** — can be decayed over training.

---

## ✅ Evaluation metrics & ideas

* **MSE** on predicted features (dx, dy, speed).
* **Dynamic Time Warping (DTW)** on 2D paths to measure shape similarity.
* Compare **speed / acceleration / curvature** distributions (histograms / KDEs).
* Visual inspection: overlay plots, animate trajectories.

---

## 🧪 Safety & ethical use

M.I.M.I.C. is intended for **research, testing, and educational** purposes only. Misuse (bypassing security, automating interactions without consent) is unethical and may be illegal. Always obtain explicit permission before testing systems that are not your own.

---

## 🧾 License

This project is released under the **MIT License**. See the `LICENSE` file for details.

---

## Appendix: example commands

Record for 2 minutes (120s):

```bash
# run main and select "Collect" (option 1), or call directly if you want
python main.py
# Then enter '1' and provide 120 when prompted
```

Train:

```bash
python main.py
# select "2" (Train model)
```

Spoof (after training):

```bash
python main.py
# select "3" (Run cursor spoofer)
```

---

## Final note

**"Code like Tony Stark, sneak like Batman."**


