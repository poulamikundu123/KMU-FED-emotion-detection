# KMU-FED Facial Emotion Recognition Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNet--B0-brightgreen.svg)](https://github.com/lukemelas/EfficientNet-PyTorch)
[![Status](https://img.shields.io/badge/Status-Phases%201--17%20Complete-success.svg)]()

An end-to-end deep learning framework for **Facial Emotion Recognition (FER)** using **Siamese Contrastive Representation Learning** and **EfficientNet-B0** on the **KMU-FED** dataset (12 subjects, 6 emotions, 1,045 processed faces).

---

## 📌 Key Highlights

- **Strict Subject-Independent Splitting:** Zero data leakage. Test subjects (02 and 11) are completely unseen during training and validation.
- **Automated Facial Preprocessing:** MTCNN face detection (10% safety margin) and affine pupil alignment (horizontal eye axis leveling).
- **Siamese Contrastive Learning:** Pairs of images trained under Euclidean contrastive loss ($m=1.0$) to cluster same-emotion facial muscle geometry and push dissimilar emotions apart.
- **High-Speed Feature Caching:** 1,280-dimensional feature vectors cached as `.npy` arrays, reducing downstream classifier training from 30 minutes to **2 seconds**.
- **Three-Stage Classifier Strategy:** Fast feature warm-up (Stage A), conservative fine-tuning guard (Stage B), and held-out evaluation (Stage C).
- **57.05% Test Accuracy on Unseen Faces:** Compared to 16.67% random guess on 6 balanced emotions.
- **Interactive Inference Tool:** Single-command emotion predictor (`predict.py`) supporting any custom image or webcam selfie with automatic face detection.

---

## 📊 Performance Scorecard (Held-Out Test Set)

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Overall Accuracy** | **57.05%** | 89 / 156 correct predictions on unseen Subjects 02 & 11 |
| **Macro Precision** | **63.82%** | Average precision across all 6 emotion categories |
| **Macro Recall** | **60.19%** | Average coverage across all 6 emotion categories |
| **Macro F1-Score** | **53.67%** | Balanced harmonic mean of Precision and Recall |
| **Weighted F1-Score**| **53.21%** | F1-Score weighted by sample frequency |

### Per-Class Performance:
| Emotion | Precision | Recall | F1-Score | Support | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Anger** | **100.00%** | 14.81% | 25.81% | 27 | 100% Precision: zero false positives |
| **Disgust** | 60.00% | 30.00% | 40.00% | 20 | Confused occasionally with Surprise |
| **Fear** | 61.76% | 53.85% | 57.53% | 39 | Strong detection coverage on subtle fear twitches |
| **Happiness** | 50.00% | **100.00%** | **66.67%** | 10 | 100% Recall: perfect catch rate |
| **Sadness** | 58.62% | **85.00%** | **69.39%** | 20 | High detection rate (17/20 correct) |
| **Surprise** | 52.54% | **77.50%** | **62.63%** | 40 | Reliable mouth/eyebrow detection |

---

## 🗂️ Project Directory Structure

```text
KMU-FED-2/
├── config.py                       # Central paths, hyperparameters, and constants
├── predict.py                      # Interactive single-image emotion inference tool
├── for_me.md                       # Beginner-friendly master guide with Q&A boxes
├── explainable.md                  # Comprehensive engineering design decisions & metrics
├── data/
│   ├── frames/                     # Verified video frames staged into 61 sequences
│   ├── detected_faces/             # Cropped faces (MTCNN with 10% margin)
│   ├── aligned_faces/              # Faces rotated to level the eye axis horizontally
│   ├── processed/                  # Final 224x224 RGB faces ready for neural networks
│   └── metadata/                   # Train/Val/Test CSVs, sequence logs, and pair datasets
├── models/
│   ├── efficientnet_encoder.py     # EfficientNet-B0 backbone (1,280 feature outputs)
│   ├── contrastive_model.py        # Siamese network with 2-layer projection head & loss
│   └── emotion_classifier.py       # MLP classifier head & EndToEndEmotionModel wrapper
├── models_checkpoints/
│   ├── best_contrastive_model.pth  # Tuned Siamese contrastive network checkpoint
│   ├── contrastive_encoder.pth     # Standalone emotion-aware feature extractor
│   ├── best_classifier_stageA.pth  # Best trained MLP emotion classifier head
│   └── best_emotion_model.pth      # Best unified end-to-end emotion model
├── features/                       # Cached 1280-dim feature matrices (.npy) & metadata
├── logs/                           # Training history CSVs and failure records
├── results/
│   └── kmu_fed/                    # Heatmaps, bar charts, 2D t-SNE/PCA plots, JSON metrics
├── preprocessing/                  # Data preparation scripts (Phases 1 to 8)
├── datasets/                       # ContrastivePairDataset with dynamic RAM caching
├── training/                       # Contrastive and classifier training pipelines
├── evaluation/                     # Metric evaluation scripts (confusion matrices & reports)
└── visualization/                  # 2D t-SNE and PCA projection scripts
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
git clone <repo-url>
cd KMU-FED-2

# Install dependencies
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn pillow mtcnn opencv-python
```

### 2. Live Interactive Prediction (Try Any Image!)
Test any face image in **0.1 seconds**:

```bash
# Run on a sample face from the dataset
python predict.py --image "data/processed/02_HA_s01/02_HA_s01_021.jpg" --save_annotated

# Or pass any image from your computer (auto-detects and crops face!)
python predict.py --image "C:\path\to\your_photo.jpg" --save_annotated
```

Sample output:
```text
==============================================================
           KMU-FED Facial Emotion Recognition Result          
==============================================================
  Input Image : data/processed/02_HA_s01/02_HA_s01_021.jpg
--------------------------------------------------------------
  >>> PREDICTED EMOTION : HAPPINESS (95.08% Confidence) <<<
--------------------------------------------------------------
  Full Class Probability Distribution:
    Anger      :   0.46%  [-------------------------]  
    Disgust    :   0.75%  [-------------------------]  
    Fear       :   3.67%  [-------------------------]  
    Happiness  :  95.08%  [#######################--] *
    Sadness    :   0.01%  [-------------------------]  
    Surprise   :   0.03%  [-------------------------]  
==============================================================
[OK] Saved annotated result image to: results/prediction_annotated.jpg
```

---

## 🛠️ Step-by-Step Pipeline Execution

| Phase | Description | Command |
| :---: | :--- | :--- |
| **1** | Inspect dataset & exclude duplicates | `python preprocessing/phase1_inspect_dataset.py` |
| **2** | Verify and stage frames into sequences | `python preprocessing/phase2_verify_frames.py` |
| **3** | Detect and crop faces using MTCNN | `python preprocessing/face_detection.py` |
| **4** | Rotate faces to level eye axis | `python preprocessing/face_alignment.py` |
| **5** | Resize to $224 \times 224$ & normalize | `python preprocessing/preprocessing.py` |
| **6** | Generate augmentation preview | `python preprocessing/augmentation.py --preview` |
| **7** | Subject-independent split (70/15/15) | `python preprocessing/split_dataset.py` |
| **8** | Generate balanced positive/negative pairs | `python preprocessing/generate_pairs.py` |
| **9** | Test contrastive dataset & DataLoader | `python datasets/contrastive_dataset.py` |
| **10**| Initialize EfficientNet-B0 encoder | `python models/efficientnet_encoder.py` |
| **11**| Test Siamese contrastive network | `python models/contrastive_model.py` |
| **12**| Train Siamese contrastive model | `python training/train_contrastive.py --epochs 3` |
| **13**| Extract & cache 1280-dim feature vectors | `python training/extract_features.py` |
| **14**| Initialize Emotion Classifier model | `python models/emotion_classifier.py` |
| **15**| Train Classifier (Three-Stage Strategy) | `python training/train_classifier.py --epochs_a 20` |
| **16**| Compute final metrics & confusion matrices | `python evaluation/evaluate.py` |
| **17**| Generate 2D t-SNE & PCA manifold plots | `python visualization/feature_visualization.py` |

---

## 📈 Visualizations & Artifacts

All figures are automatically generated in [`results/kmu_fed/`](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/results/kmu_fed):

- `confusion_matrix.png`: Heatmap of raw prediction counts per emotion.
- `confusion_matrix_normalized.png`: Percentage recall heatmap.
- `per_class_metrics.png`: Grouped bar chart comparing Precision, Recall, and F1 per class.
- `tsne_features_test.png`: 2D t-SNE scatter plot of test faces colored by emotion.
- `tsne_features_all.png`: 2D t-SNE scatter plot across all 1,045 KMU-FED faces.
- `pca_features_test.png`: 2D PCA projection of the test set feature space.
- `clustering_metrics.json`: Quantitative cluster metrics (Silhouette: `0.0483`, Davies-Bouldin: `2.3438`).

---

## 📖 In-Depth Documentation

- [**`for_me.md`**](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/for_me.md): Intuitive, simple-English walkthrough with real-world analogies and highlighted Q&A boxes for all common questions.
- [**`explainable.md`**](file:///c:/Users/kundu/OneDrive/Desktop/KMU-FED-2/explainable.md): Complete engineering documentation with mathematical proofs, ablation studies, and architectural rationales.
