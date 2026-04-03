# Verification-aware Speech Recognition
A case study of CNN-based speech recognition evaluated with neural network verifiers for robustness analysis. This repository contains code associated with the paper:  
**Verification-Aware Convolution Neural Networks for Speech Recognition: A Case Study**,   
*Syed Ali Asadullah Bukhari, Barak A. Pearlmutter, and Rosemary Monahan*  *Formal Methods in Software Engineering (Formalise'26), 2026 (To Appear)*, DOI: https://doi.org/10.1145/3793656.3793683

---

## Overview

Speech recognition systems are often evaluated based on accuracy alone. In this project, we use **neural network verification techniques** to assess the **robustness** of CNN-based models applied to both raw audio signals and spectrogram representation.

This repository demonstrates:
- A complete **audio preprocessing pipeline**
- **CNN-based speech recognition model training**
- Application of **neural network verification tools** to analyze model behavior for robustness.

---

## Pipeline Overview

The project follows a sequential pipeline:

### 1. Data Preparation
- `00_dataset.py` – Initial dataset preparation  
- `00_dataset_1_sec.py` – Audio segmentation into 1-second clips  
- `00_dataset_downsample.py` – Downsampling audio signals  
- `00_dataset_spectrogram.py` – Conversion to spectrograms  

### 2. Model Training
- `01_training.py` – Training CNN-based speech recognition models

### 3. Verification
- `TBA` – TBA 

---

## How to Run

Execute the scripts in the following order:

```bash
python 00_dataset.py
python 00_dataset_1_sec.py
python 00_dataset_downsample.py
python 00_dataset_spectrogram.py
python 01_training.py
