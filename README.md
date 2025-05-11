# Hybrid Optimization of NLP and Vision Models using RKD and QLoRA

This repository presents a high-performance machine learning project that compresses large Transformer models (BERT for NLP and ViT for Vision) using a hybrid pipeline that combines **Recursive Knowledge Distillation (RKD)**, **Multi-Agent KD**, **QLoRA (Quantized Low-Rank Adaptation)**, and **Pruning** techniques. The resulting models retain strong performance while being highly efficient and suitable for deployment on mobile and edge devices.

## 🔍 Project Overview

Modern Transformer models achieve excellent accuracy but are often too large for real-time or edge deployment. This project demonstrates how to systematically reduce model size while preserving accuracy by applying the following:

* **Recursive Knowledge Distillation (RKD):** Multi-stage teacher-student distillation
* **Multi-Agent KD:** Uses ensemble knowledge from multiple teachers
* **QLoRA:** Combines 4-bit quantization with trainable LoRA adapters
* **Pruning:** L1 unstructured pruning to reduce parameters post-finetuning

---


## 📁 Repository Structure

```
├── rkd+qlora_bert.ipynb         # Full notebook for BERT-based NLP compression (SST-2)
├── rkd+qlora_vit.ipynb          # Full notebook for ViT-based Vision compression (Beans dataset)
├── README.md                    # Project overview and usage instructions
```

---

## 📊 Datasets and Tasks

* **BERT (NLP):** [GLUE SST-2 Dataset](https://gluebenchmark.com/tasks) – Binary sentiment classification
* **ViT (Vision):** [Beans Image Classification Dataset](https://huggingface.co/datasets/beans) – 3-class crop disease detection

---

## ✅ Milestones & Completion Status

| Milestone                         | Status     |
| --------------------------------- | ---------- |
| Data preprocessing & loading      | ✅ Complete |
| Model training (BERT & ViT)       | ✅ Complete |
| Knowledge Distillation strategies | ✅ Complete |
| QLoRA integration                 | ✅ Complete |
| Pruning integration               | ✅ Complete |
| Results evaluation + plots        | ✅ Complete |

---

## 🚀 How to Run the Code

1. Clone the repository or open the `.ipynb` notebooks on Google Colab.
2. Install the required dependencies:

```bash
pip install transformers datasets accelerate evaluate peft bitsandbytes
```

3. Execute the notebooks:

   * `rkd+qlora_bert.ipynb` for NLP
   * `rkd+qlora_vit.ipynb` for Vision

Each notebook is modular, includes KD training loops, QLoRA integration, pruning routines, and plots.

---

## 📈 Results Summary

### 🔤 BERT Results (SST-2)

| Method                | Mini-BERT Accuracy | Params |
| --------------------- | ------------------ | ------ |
| Base Model            | 91.7%              | 110M   |
| Straightforward KD    | 86.6%              | 11M    |
| Successive KD         | 87.3%              | 11M    |
| Multi-Agent KD        | 87.3%              | 11M    |
| QLoRA + Pruning (30%) | 85.1%              | 11M ↓  |

### 🖼️ ViT Results (Beans Dataset)

| Model                     | Parameters | Accuracy |
| ------------------------- | ---------- | -------- |
| ViT-12                    | 86M        | 93.8%    |
| ViT-9                     | 64M        | 83.6%    |
| ViT-6                     | 43M        | 81.3%    |
| ViT-3                     | 22M        | 82.0%    |
| Final ViT-3 (QLoRA+Prune) | 22M ↓      | 93.8% ↑    |

---

## 📚 References

1. Pagnoni et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
2. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
4. Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)

---

## 👥 Authors

* Amaan Mithani
* Kunaal Vadgama

NYU | High Performance Machine Learning (Spring 2025)
