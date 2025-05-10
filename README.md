# ⚡ Electric Fault Detection & Classification System

A machine learning-powered application for detecting and classifying electrical faults in three-phase power systems, featuring an interactive Streamlit interface and real-time waveform visualization.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Details](#model-details)


---

## 📝 Overview

This project leverages machine learning (Random Forest) to automatically detect and classify faults in three-phase electrical power systems. The system takes current and voltage measurements as input and provides both binary (fault/no fault) and multiclass (fault type) predictions, helping engineers and operators quickly identify and respond to abnormal conditions.

---

## ✨ Features

- **Binary Fault Detection:** Instantly determines if a fault is present.
- **Multiclass Fault Classification:** Identifies the specific type of fault:
  - Single Line-to-Ground (LG)
  - Line-to-Line (LL)
  - Line-to-Line-to-Ground (LLG)
  - Line-to-Line-to-Line (LLL)
- **Interactive UI:** Built with Streamlit for easy data entry and visualization.
- **Sample Data Loader:** Quickly test the system with realistic sample scenarios.
- **Real-Time Visualization:** View three-phase current waveforms based on your inputs.


## 🖥️ Usage

1. **Input Measurements:** Enter or load sample current and voltage values for each phase.
2. **Analyze:** Click "Analyze System" to run the models.
3. **View Results:** See binary and multiclass predictions, recommendations, and waveform plots.
4. **Interpret:** Use the provided recommendations to guide your response.

---

## 🤖 Model Details

- **Binary Model:** XGBoost classifier for fault/no fault detection.
- **Multiclass Model:** XGBoost classifier for fault type classification.
- **Scaler:** MinMaxScaler for feature normalization.
- **Inputs:** Phase currents (Ia, Ib, Ic) and voltages (Va, Vb, Vc).
- **Outputs:** Fault status and type.

