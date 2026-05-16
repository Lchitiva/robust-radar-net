# 📡 RobustRadarNet — FMCW Radar Clutter Removal via Binary Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Domain](https://img.shields.io/badge/Domain-Radar%20Signal%20Processing-blueviolet)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> A neural network that learns to separate real hand gesture detections from radar clutter — making downstream gesture recognition dramatically more robust.

*Companion project to [Radar Hand Digit Recognition](../radar-digit-recognition/) · Ruhr-Universität Bochum*

---

## 🎯 Problem Statement

FMCW radars use the **CFAR (Constant False Alarm Rate)** algorithm to detect objects from raw radar signals. CFAR is powerful but indiscriminate: it triggers on *anything* that exceeds an adaptive threshold — including reflections from clothing, sensor housing, and background objects. These unwanted detections are called **clutter**.

```
Raw radar signal → CFAR → Detections (gesture + clutter mixed)
                                           ↓
                                   [RobustRadarNet]
                                           ↓
                              Clean detections (clutter removed)
```

Without clutter removal, downstream classifiers receive noisy, inconsistent input. This project trains a neural network to solve the clutter problem at the source.

### Why Not a Simple Threshold?

A naive range threshold (e.g., "discard anything beyond 0.4 m") fails because **clutter and gesture detections overlap in range-velocity space**. The distinguishing factor is **temporal context**: real gestures produce consistent, coherent detection trajectories across frames; clutter appears sporadically in isolated frames with no consistent pattern. RobustRadarNet exploits this by encoding a ±3 frame temporal window around each detection.

---

## 🏗️ Architecture

### Input Features

Each CFAR detection is classified using a temporal context window of **±3 frames** (7 frames total). For each frame in the window, features from both sensors are extracted:

| Feature group | Description |
|--------------|-------------|
| Range (S0, S1) | Distance of detection from sensor |
| Velocity (S0, S1) | Doppler-derived radial velocity |
| Detection count | Number of detections in that frame |
| Temporal position | Normalized frame index |

Total input dimensionality: **7 frames × 6 features = 42 features**

### Model: Temporal MLP

```
Input (42)  →  FC(128) + BatchNorm + ReLU + Dropout(0.3)
            →  FC(64)  + BatchNorm + ReLU + Dropout(0.3)
            →  FC(32)  + ReLU
            →  FC(1)   + Sigmoid
            →  Binary output: 1 = gesture, 0 = clutter
```

### Label Generation

Ground truth labels are generated automatically by comparing CFAR detections against ground truth trajectories: a detection is labeled **gesture (1)** if it lies within 0.02 m of the nearest ground truth point in the same frame, and **clutter (0)** otherwise.

---

## 📊 Results

### Dataset Statistics

The EDA reveals the scope of the clutter problem:

| Statistic | Value |
|-----------|-------|
| Average clutter rate | ~60–70% of detections are clutter |
| Max clutter rate (single trial) | >90% |
| Trials with >10% clutter | majority |

This confirms that **raw CFAR output is too noisy to use directly** for gesture recognition.

### Model Performance

Evaluated using **person-independent splits** (GroupShuffleSplit by person ID):

| Metric | Value |
|--------|-------|
| AUC-ROC | reported in notebook |
| Average Precision | reported in notebook |
| F1 Score | reported in notebook |
| Precision (gesture) | reported in notebook |
| Recall (gesture) | reported in notebook |

> The notebook outputs a full ROC curve, Precision-Recall curve, and confusion matrix. High recall (catching most real gesture detections) is prioritized over precision — a missed gesture point is worse than a surviving clutter point for downstream classification.

---

## 🔬 Exploratory Data Analysis

The EDA is designed to make the clutter problem visually undeniable:

**Figure 1 — The Clutter Problem**
Side-by-side visualization of: (a) raw CFAR output colored by sensor, (b) same detections colored by label (gesture = blue, clutter = red), (c) ground truth trajectory. This makes the challenge immediately obvious.

**Figure 2 — Clutter Distribution**
- Histogram of clutter rates across all ~1,000 trials
- Average clutter rate broken down by gesture type — some gestures (like '1') produce more clutter than others (like '0')
- Average clutter rate broken down by person — individual anatomy and writing style affect clutter levels

**Figure 3 — Temporal Consistency**
A frame-by-frame animation of detection counts showing how gesture detections cluster in time while clutter appears randomly — the physical intuition behind the ±3 frame window feature.

---

## 🛠️ Tech Stack

- **PyTorch** — MLP model, training loop, binary cross-entropy with pos_weight for class imbalance
- **Scikit-learn** — GroupShuffleSplit, StandardScaler, ROC/PR curves
- **NumPy** — feature extraction, temporal windowing
- **Pandas** — clutter statistics aggregation
- **Matplotlib / Seaborn** — all visualizations (clutter problem figures, training curves, ROC/PR curves)
- **tqdm** — progress tracking during dataset construction

---

## 🚀 How to Run

### Prerequisites

The dataset (`Detections_Dataset.zip`) must be stored in your Google Drive:

```
MyDrive/
└── Detections_Dataset.zip
    └── CFAR_detections/
    └── GroundTruth_detections/
```

### Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lchitiva/robust-radar-net/blob/main/RobustRadarNet.ipynb)

1. Mount Google Drive when prompted
2. Adjust `ZIP_PATH` in Part 0 if your Drive path differs
3. Run all cells sequentially

---

## 📁 Project Structure

```
robust-radar-net/
├── RobustRadarNet.ipynb   # Full notebook: EDA + feature eng. + model + evaluation
└── README.md
```

*Note: The dataset is not included in this repository (proprietary research data). Contact the Ruhr-Universität Bochum Praxis project supervisors for access.*

---

## 🔗 Relationship to Digit Recognition Project

This project and [Radar Hand Digit Recognition](../radar-digit-recognition/) form a **two-stage pipeline**:

```
Stage 1 — RobustRadarNet     : Remove clutter from CFAR detections
Stage 2 — Digit Recognition  : Classify the cleaned gesture trajectory
```

Running Stage 1 first significantly improves Stage 2 accuracy by providing cleaner, more consistent input sequences to the CNN + BiLSTM classifier.

---

## 💡 Key Takeaways

1. **Clutter is not random** — it follows physical patterns (certain gestures and certain people produce more clutter) that a model can learn
2. **Temporal context is the key feature** — single-frame features cannot distinguish clutter from gesture; the ±3 frame window is what makes the model work
3. **Class imbalance matters** — with ~65% clutter, a naive model would predict "gesture" everywhere and score well on accuracy. Using `pos_weight` in BCELoss and monitoring F1/AUC instead of accuracy is essential
4. **Person-independent evaluation** is required — clutter patterns vary by person, so random splits would leak person-specific information into the test set
5. **Pre-processing as ML** — this project demonstrates that signal cleaning can itself be learned, enabling end-to-end trainable radar processing pipelines

---

## 📚 References

- Richards, M.A. (2005). *Fundamentals of Radar Signal Processing*. McGraw-Hill.
- Lien, J. et al. (2016). *Soli: Ubiquitous Gesture Sensing with Millimeter Wave Radar*. ACM SIGGRAPH.
- 2pi-Labs SENSE X1155S FMCW Radar Sensor — [2pi-labs.com](https://2pi-labs.com)
- Ruhr-Universität Bochum — Praxis Project, Digital Signal Processing Group
