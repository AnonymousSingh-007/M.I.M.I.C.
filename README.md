# M.I.M.I.C.
**Motion Imitation Mechanism for Input Camouflage**  
*Looks real. Isn’t.*

---

## Summary

M.I.M.I.C. is a machine learning–based cursor spoofer that replicates human-like input patterns for use in adversarial testing, sandbox evasion, and behavioral spoofing.

By recording real user mouse trajectories and training predictive models on the temporal and spatial patterns, M.I.M.I.C. generates synthetic motion indistinguishable from legitimate interaction—at least to conventional detection systems.

---

## Core Objectives

- Record native human mouse input with timestamps
- Train a prediction model to imitate behavioral patterns
- Simulate cursor movements using model outputs
- Evaluate the resilience of detection systems to input-based spoofing

---

## Features

| Feature                                         | Status |
|--------------------------------------------------|-------  |
| Mouse movement capture with timestamp            | ✅     |
| CSV export for reproducibility                   | ✅     |
| MLP model training on user data (PyTorch)        | ✅     |
| Cursor spoofing via trained model                | ✅     |
| Smart spoof duration (≤ 75% of real trace time)  | ✅     |
| Console interface and motion visuals             | ✅     |

---

## Installation

Python 3.13.5 is recommended. Create a virtual environment to avoid conflicts.

## How It Works
# Data Capture
--Records mouse trajectory every ~10ms
--Saves absolute screen coordinates with high precision

# Model Architecture
--Simple MLP (Multi-Layer Perceptron)
--Input: relative timestamp (float)
--Output: predicted (x, y) screen coordinates
--Trained using MSE loss

# Cursor Simulation
--Spoofs path over selected duration (capped at 75% of original)
--Interpolates micro-movements for realism
--Optional idle jitter and brief hesitations

```bash


git clone https://github.com/<your-org>/mimic-cursor-spoofer.git
cd mimic-cursor-spoofer
pip install -r requirements.txt
