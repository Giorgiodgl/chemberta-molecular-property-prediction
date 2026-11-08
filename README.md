# 🧪 ChemBERTa Multi-Task Molecular Property Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)

End-to-end machine learning pipeline fine-tuning a pre-trained chemical language model (`seyonec/PubChem10M_SMILES_BPE_450k`) on the MoleculeNet **Tox21** benchmark dataset for multi-task molecular property prediction.

---

## 🎯 Project Overview
Predicting biological assay outcomes and toxicity from raw molecular structures is a core challenge in early-stage drug discovery. This repository implements a Hugging Face `Transformers` and PyTorch pipeline to process SMILES strings directly, bypassing traditional handcrafted molecular fingerprints.

---

## 📊 Training & Performance Results
Trained on Google Colab (GPU-accelerated) with automated macro-averaged **ROC-AUC** evaluation across multi-label toxicity targets.

| Epoch | Training Loss | Validation Loss | ROC-AUC |
| :---: | :---: | :---: | :---: |
| **1** | 0.2075 | 0.1989 | 0.7427 |
| **2** | 0.1835 | 0.1844 | 0.7944 |
| **3** | **0.1666** | **0.1808** | **0.8033** |

---

## 🔍 Scope & Engineering Architecture
This project is structured as a **technical proof-of-concept pipeline** demonstrating clean MLOps integration, custom PyTorch data wrappers, and metric evaluation. 

While the baseline run successfully achieves a competitive **0.803 ROC-AUC** within a rapid training cycle, a fully optimized production deployment would scale to include:
* **Extended Training & Early Stopping:** Monitoring validation loss plateaus across 10–20+ epochs.
* **K-Fold Cross-Validation:** Replacing a single train/test split with multi-fold validation to guarantee structural generalization across chemical scaffolds.
* **Hyperparameter Sweeps:** Systematically tuning learning rate schedules, warmups, and dropout rates.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Hugging Face `Transformers` (`Trainer`, `AutoModelForSequenceClassification`), `Datasets`
* **Cheminformatics & Metrics:** SMILES string parsing, NumPy, Pandas, Scikit-Learn (`roc_auc_score`)
* **Environment:** Google Colab (T4 GPU execution)

---

## 📂 Repository Structure
```text
├── chemberta_molecular_property_prediction.ipynb   # Main Google Colab execution notebook
├── inference.py                                    # Inference script for evaluating custom SMILES
├── requirements.txt                                # Project dependencies
└── README.md                                       # Project documentation
