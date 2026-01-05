

# Intrusion Detection using Granular-Ball Intuitionistic Fuzzy Sets (GBIFS)

[![Python](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An intrusion detection framework based on **Granular-Ball Intuitionistic Fuzzy Sets (GBIFS)**, designed to enhance robustness, accuracy, and interpretability for heterogeneous and sparse IIoT and network traffic data.

---

## 🔍 Overview

With the deep integration of the **Industrial Internet of Things (IIoT)** into critical infrastructures such as intelligent manufacturing and energy management, cybersecurity threats have become increasingly complex and diverse.
Traditional intrusion detection methods often struggle with:

* Heterogeneous feature distributions
* Sparse and imbalanced intrusion patterns
* High-dimensional and redundant data
* Limited interpretability of detection results

To address these challenges, this project implements an intrusion detection framework based on **Granular-Ball Intuitionistic Fuzzy Sets (GBIFS)**, which tightly integrates:

* **Granular-Balls (GB)** for adaptive, class-wise data granulation
* **Intuitionistic Fuzzy Sets (IFS)** to model uncertainty and hesitation in intrusion patterns
* An improved intuitionistic fuzzy distance metric for precise classification

The framework constructs **Granular-Ball Intuitionistic Fuzzy Patterns (GBIFP)** that conform to the intrinsic feature distributions of IIoT data, enabling effective discrimination between normal traffic and diverse attack behaviors.

> 📌 **Key Insight**
> By combining adaptive multi-granularity representation with intuitionistic fuzzy reasoning, the GBIFS framework overcomes the limitations of radius-dependent granulation and achieves robust performance under heterogeneous and sparse data conditions.

---

## ✨ Main Contributions

* A **novel class-wise Granular-Ball generation strategy** tailored to IIoT intrusion data
* Construction of **Granular-Ball Intuitionistic Fuzzy Patterns (GBIFP)** aligned with feature distributions
* An **improved intuitionistic fuzzy distance metric** for accurate intrusion classification
* Superior performance validated on both **IIoT datasets** (X-IIoTID, TON-IOT, WUSTL-IIOT) and **classical NIDS datasets** (KDDCUP99, NSL-KDD, UNSW-NB15)

---

## 📂 Project Structure

```
├── CWGB.py              # Construction of class-discriminative Granular-Balls
├── GBIFSmodel.py        # Core implementation of the GBIFS model
├── preprocess_data.py   # Data loading, normalization, and splitting
├── metric.py            # Custom metrics (e.g., Accuracy, Precision, Recall, F1_score, FPR)
├── data/                # Place for sample datasets or links
├── download_data.py     # download dataset form Google Drive
└── results/             # Saved outputs, logs, figures
```

---

## 🚀 Quick Start

### Environment

* **Python**: 3.10.12
* **NumPy**: 1.26.4
* **Pandas**: 2.1.4
* **scikit-learn**: 1.3.0
* * **gdown**: 5.2.0

### Installation

use the provided `requirements.txt` (see below):

```bash
pip install -r requirements.txt
```

### Run


Download and prepare the preprocessed dataset:
```bash
python download_data.py
```
This will:
Download the 2GB dataset from Google Drive
Extract all files into ./data

Run the main experiment:
```bash
python GBIFSmodel.py 
```

---

## 📊 Evaluation Metrics

The framework supports multiple evaluation metrics tailored for imbalanced intrusion detection tasks (see `metric.py`):

* Accuracy
* Precision
* Recall
* F1-score
* False Positive Rate (FPR)


---

## 📄 License

MIT License — see LICENSE for details.

