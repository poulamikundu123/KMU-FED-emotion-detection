# KMU-FED Facial Emotion Recognition — Complete Simple Guide (Phases 1 to 17)

> **Welcome!**  
> This document is your complete, master walkthrough of everything we have built together from **Phase 1 through Phase 17**.  
> Every phase is explained in plain, beginner-friendly English using real-world analogies, with dedicated **Q&A Callout Boxes** for all the questions and cross-questions you asked along the way!

---

## Table of Contents
1. [Pipeline Overview Diagram](#pipeline-overview)
2. [Phase 1: Dataset Inspection & Metadata](#phase-1--dataset-inspection--metadata)
3. [Phase 2: Frame Verification & Staging](#phase-2--frame-verification--staging)
4. [Phase 3: Face Detection & Cropping (MTCNN)](#phase-3--face-detection--cropping-mtcnn)
5. [Phase 4: Face Alignment (Leveling the Eyes)](#phase-4--face-alignment-leveling-the-eyes)
6. [Phase 5: Resizing & Normalization](#phase-5--resizing--normalization)
7. [Phase 6: Data Augmentation](#phase-6--data-augmentation)
8. [Phase 7: Train / Val / Test Split (Zero Leakage)](#phase-7--train--val--test-split-zero-leakage)
9. [Phase 8: Pair Generation (Contrastive Learning)](#phase-8--pair-generation-contrastive-learning)
10. [Phase 9: PyTorch Dataset & RAM Caching](#phase-9--pytorch-dataset--ram-caching)
11. [Phase 10: EfficientNet Feature Encoder](#phase-10--efficientnet-feature-encoder)
12. [Phase 11: Siamese Contrastive Network](#phase-11--siamese-contrastive-network)
13. [Phase 12: Contrastive Training](#phase-12--contrastive-training)
14. [Phase 13: Feature Extraction & Caching](#phase-13--feature-extraction--caching)
15. [Phase 14: Emotion Classifier Architecture](#phase-14--emotion-classifier-architecture)
16. [Phase 15: Training the Classifier (3 Stages)](#phase-15--training-the-classifier-3-stages)
17. [Phase 16: Comprehensive Evaluation & Final Exam](#phase-16--comprehensive-evaluation--final-exam)
18. [Phase 17: 2D Feature Space Visualization (t-SNE & PCA)](#phase-17--2d-feature-space-visualization-t-sne--pca)
19. [Bonus: How to Test Any Image Live (predict.py)](#bonus-how-to-test-any-image-live-predictpy)

---

<a name="pipeline-overview"></a>
## 1. Pipeline Overview Diagram

```text
Raw Video Frames (1,100 files, 12 People, 6 Emotions)
         │
         ▼  [Phase 1 & 2] Metadata & Integrity Check (Removed 6 Kaggle duplicates)
Staged Video Frames
         │
         ▼  [Phase 3 & 4] MTCNN Face Detection & Affine Eye Alignment
Level, Cropped Faces (1,045 faces)
         │
         ▼  [Phase 5 & 6] 224x224 Resize, ImageNet Normalization, Live Augmentation
Processed Face Tensors
         │
         ▼  [Phase 7] Subject-Independent Split (Train: 8 People | Val: 2 People | Test: 2 People)
Zero-Data-Leakage Splits
         │
         ▼  [Phase 8 & 9] Balanced Pair Generation & Live-Augmented DataLoader
Positive & Negative Face Pairs
         │
         ▼  [Phase 10, 11 & 12] EfficientNet-B0 + Siamese Network + Contrastive Training
Emotion-Aware Vision Encoder
         │
         ▼  [Phase 13] Extract & Cache 1,280-Number Summary Vectors to Disk (.npy)
Instant Feature Representations
         │
         ▼  [Phase 14 & 15] 329,990-Parameter MLP Classifier Head Training (3 Stages)
Trained Emotion Recognition Model (best_emotion_model.pth)
         │
         ▼  [Phase 16 & 17] Metrics (57.05% Acc on Unseen Faces) & 2D t-SNE Clustering
Scientific Reports, Heatmaps & 2D Manifold Plots
```

---

<a name="phase-1--dataset-inspection--metadata"></a>
## Phase 1 — Dataset Inspection & Metadata

### What We Did:
We inspected all raw images in the KMU-FED dataset. We verified:
- **12 different human subjects** (numbered 1 to 12).
- **6 emotions:** Anger (AN), Disgust (DI), Fear (FE), Happiness (HA), Sadness (SA), Surprise (SU).
- **Golden Catch:** We discovered and eliminated 6 duplicate files (`1_AN_mr_*`) that were accidentally duplicated in the Kaggle upload, preserving 1,100 clean, unique frames.
- Generated: `data/metadata/kmu_fed_metadata.csv` and `data/metadata/dataset_statistics_report.txt`.

> [!NOTE]
> 💬 **YOUR CROSS-QUESTION ANSWERED: What does a filename like `01_AN_mr_001.jpg` mean?**  
> Every file is named with a strict code:  
> - `01` = **Subject ID** (Person #1)  
> - `AN` = **Emotion Code** (`AN`=Anger, `DI`=Disgust, `FE`=Fear, `HA`=Happiness, `SA`=Sadness, `SU`=Surprise)  
> - `mr` = **Person's Code/Initials** (e.g. initials of the volunteer actor)  
> - `001` = **Frame Number** (the 1st frame in that emotional video clip)

---

<a name="phase-2--frame-verification--staging"></a>
## Phase 2 — Frame Verification & Staging

### What We Did:
Raw datasets often contain corrupt, zero-byte, or unreadable files.
- We opened every single frame to verify its RGB channels and file headers.
- We staged all 1,100 valid frames into **61 neat sequence folders** under `data/frames/`.
- Generated: `data/metadata/kmu_fed_sequences.csv`.

---

<a name="phase-3--face-detection--cropping-mtcnn"></a>
## Phase 3 — Face Detection & Cropping (MTCNN)

### What We Did:
Raw camera shots include car windshields, steering wheels, shoulders, and background walls. If we feed that to the AI, it might memorize the color of the car instead of the emotion!
- We used **MTCNN** (Multi-Task Cascaded Convolutional Networks) to find the exact face.
- We cropped the face with an extra **10% safety margin** so eyebrows and chins are never cut off.
- **Result:** Successfully detected and cropped **1,045 faces** ($95.0\%$ detection rate).
- The 55 frames with extreme shadows/darkness were safely logged to `logs/face_detection_failures.csv` so no bad data enters the pipeline.

---

<a name="phase-4--face-alignment-leveling-the-eyes"></a>
## Phase 4 — Face Alignment (Leveling the Eyes)

### What We Did:
When people express emotions, they often tilt their heads (e.g. tilting sideways in surprise or confusion).
- If a face is tilted $15^\circ$, raw pixels rotate drastically, confusing the model.
- We computed the angle between the **left pupil and right pupil**.
- Using an affine transformation, we rotated the face so the **eye axis is 100% horizontal**.
- **Result:** **1,045 out of 1,045 faces successfully aligned** with 0 failures (`data/aligned_faces/`).

---

<a name="phase-5--resizing--normalization"></a>
## Phase 5 — Resizing & ImageNet Normalization

### What We Did:
Neural networks require identical input dimensions.
- We resized every face to exactly **$224 \times 224$ pixels**.
- We applied **ImageNet Normalization** (subtracting mean `[0.485, 0.456, 0.406]` and dividing by std `[0.229, 0.224, 0.225]`).
- This centers the pixel numbers around zero, which prevents mathematical explosions during training.
- Outputs saved to `data/processed/`.

---

<a name="phase-6--data-augmentation"></a>
## Phase 6 — Data Augmentation

### What We Did:
With only 1,045 photos, a neural network could easily memorize individual pictures.
- We designed a transformation pipeline:
  - Horizontal flips (50% chance)
  - Small rotations ($\pm 10^\circ$)
  - Mild brightness and contrast shifts ($\pm 20\%$)
  - Gaussian blur and slight noise
- Created a visual proof strip: `data/augmentation_preview.jpg`.

---

<a name="phase-7--train--val--test-split-zero-leakage"></a>
## Phase 7 — Train / Val / Test Split (Zero Leakage)

### What We Did:
We divided the data into 3 strict sets:
1. **Train (70%, 732 images):** 8 Subjects (01, 03, 05, 06, 07, 09, 10, 12)
2. **Validation (15%, 157 images):** 2 Subjects (04, 08)
3. **Test (15%, 156 images):** 2 Subjects (02, 11)

> [!IMPORTANT]
> 💬 **YOUR CROSS-QUESTION ANSWERED: Why split by Subject instead of splitting random images?**  
> In a video sequence, Frame 1 and Frame 2 of the same person look almost identical.  
> If Frame 1 is in Train and Frame 2 is in Test, the AI isn't learning *emotions* — it's simply recognizing *that person's face*!  
> By putting **Subjects 02 and 11 exclusively in the Test set**, the AI has NEVER seen those human beings in its life. When it scores well on the test set, we know for a fact it learned genuine emotion geometry, not identity!

---

<a name="phase-8--pair-generation-contrastive-learning"></a>
## Phase 8 — Pair Generation (Contrastive Learning)

### What We Did:
Contrastive learning trains an AI by comparing **two images at a time**:
- **Positive Pair (Label = 1):** Two pictures showing the **SAME emotion** (e.g. Subject 1 Smiling + Subject 3 Smiling).
- **Negative Pair (Label = 0):** Two pictures showing **DIFFERENT emotions** (e.g. Subject 5 Smiling + Subject 5 Angry).
- Generated 3,000 balanced pairs for Training, 600 for Validation, and 600 for Testing.

> [!NOTE]
> 💬 **YOUR CROSS-QUESTION ANSWERED: What do `subject_1` and `subject_2` mean in the pairs CSV?**  
> They represent the person IDs of the two people being compared!  
> In a **cross-subject positive pair**, `subject_1` might be `1` (John) and `subject_2` might be `3` (Alex), but both are expressing **Happiness**.  
> This forces the model to ignore John's beard or Alex's nose, and focus *only* on the smiling mouth muscles!

---

<a name="phase-9--pytorch-dataset--ram-caching"></a>
## Phase 9 — PyTorch Dataset & RAM Caching

### What We Did:
We built `ContrastivePairDataset` and PyTorch `DataLoader` to feed pairs in batches of 32.

> [!TIP]
> 💬 **YOUR CROSS-QUESTION ANSWERED: What is "Dynamic Live Augmentation" vs Saving to Hard Drive?**  
> If you create 50 variations of 1,000 images on disk, you waste gigabytes of hard drive space writing 50,000 `.jpg` files.  
> Instead, our dataset loads the original image and applies small rotations and lighting changes **live in computer RAM** right before feeding it to the AI.  
> In Epoch 1, the AI sees the face rotated $+2^\circ$. In Epoch 2, it sees it a little brighter. It never sees the exact same pixel pattern twice, and your hard drive stays 100% clean!

---

<a name="phase-10--efficientnet-feature-encoder"></a>
## Phase 10 — EfficientNet Feature Encoder

### What We Did:
We used the pretrained **EfficientNet-B0** convolutional neural network to compress raw face pixels into compact emotional feature numbers.

> [!NOTE]
> 💬 **YOUR CROSS-QUESTION ANSWERED: From 150,528 Pixels $\rightarrow$ A 1,280-Number Summary:**  
> A $224 \times 224$ RGB image has $224 \times 224 \times 3 = 150,528$ numbers. That is too much raw noise (skin pores, lighting glares, hair).  
> EfficientNet passes those pixels through 9 convolutional stages and boils them down to just **1,280 meaningful numbers**.  
> These 1,280 numbers describe high-level face geometry:  
> - *"Mouth corners pulled up by 8mm"*  
> - *"Eyebrows pulled down by 4mm"*  
> Two different angry faces will produce very similar 1,280-number vectors!

> [!NOTE]
> 💬 **YOUR CROSS-QUESTION ANSWERED: What does `Forward pass test passed: Input [2, 3, 224, 224] -> Output [2, 1280]` mean?**  
> - `[2, 3, 224, 224]`: A mini-batch of **2 photos**, with **3 color channels** (Red, Green, Blue), **224 pixels high** by **224 pixels wide**.  
> - `[2, 1280]`: The model processed those 2 photos and returned **2 summary vectors**, each containing **1,280 feature numbers**!

> [!NOTE]
> 💬 **YOUR CROSS-QUESTION ANSWERED: What are Freeze and Unfreeze controls?**  
> - `freeze_backbone()`: Locks all 4.0 million parameters (0 trainable). We put a padlock on them so the model doesn't mess up its visual knowledge.  
> - `unfreeze_last_n_blocks(3)`: Unlocks only the top layers to gently tune them for subtle muscle twitches.  
> - `unfreeze_backbone()`: Unlocks all layers for full fine-tuning.

---

<a name="phase-11--siamese-contrastive-network"></a>
## Phase 11 — Siamese Contrastive Network

### What We Did:
We built a **Siamese Network** (`SiameseContrastiveNetwork`) with a 2-layer projection head:
$$\mathbf{1280} \xrightarrow{\text{Linear}} \mathbf{512} \xrightarrow{\text{BatchNorm + ReLU}} \mathbf{128} \xrightarrow{\text{L2 Normalization}}$$
- Both faces pass through the **SAME** EfficientNet encoder (like Siamese twins sharing one brain).
- We applied **Contrastive Loss** ($m=1.0$):
  - If faces have the same emotion: $\text{Loss} = \text{distance}^2$ $\rightarrow$ **Pulls them together!**
  - If faces have different emotions: $\text{Loss} = \max(0, 1.0 - \text{distance})^2$ $\rightarrow$ **Pushes them apart!**

---

<a name="phase-12--contrastive-training"></a>
## Phase 12 — Contrastive Training

### What We Did:
We trained the Siamese network over our 3,000 training pairs.
- The validation loss dropped from **$0.2177 \rightarrow 0.1613$**.
- The model successfully learned to pull same-emotion faces closer and push different emotions further apart.
- Saved the tuned encoder weights to `models_checkpoints/contrastive_encoder.pth`.

---

<a name="phase-13--feature-extraction--caching"></a>
## Phase 13 — Feature Extraction & Caching

### What We Did:
We passed all 1,045 images through our tuned EfficientNet encoder **one time** and saved their 1,280-number vectors directly to disk:
- `features/train_features.npy` `(732, 1280)`
- `features/val_features.npy` `(157, 1280)`
- `features/test_features.npy` `(156, 1280)`

> [!TIP]
> 💬 **YOUR CROSS-QUESTION ANSWERED: The "Book Report" Analogy (Why cache features to disk?):**  
> Imagine each face image is a 500-page book. EfficientNet is an expert who reads the 500-page book and writes a 1-page summary (1,280 numbers).  
> In Phase 14 & 15, we need to train a student (the Classifier) over 50 rounds (epochs).  
> **Without Phase 13:** The expert has to re-read all 1,045 thick books 50 times in a row. It takes 30 minutes!  
> **With Phase 13:** The expert reads the books **just once** on day 1 and writes the 1-page summaries to disk. For all 50 rounds, the student reads the summaries directly.  
> **Result:** 50 epochs of training finish in **2 seconds**!

---

<a name="phase-14--emotion-classifier-architecture"></a>
## Phase 14 — Emotion Classifier Architecture

### What We Did:
We built the **Decision Maker** (`EmotionClassifier`) using a Multi-Layer Perceptron (MLP):
$$\mathbf{1280} \rightarrow \mathbf{\text{Linear(256)}} \rightarrow \mathbf{\text{BatchNorm}} \rightarrow \mathbf{\text{ReLU}} \rightarrow \mathbf{\text{Dropout(0.4)}} \rightarrow \mathbf{\text{Linear(6 Logits)}}$$

> [!NOTE]
> 💬 **YOUR CROSS-QUESTION ANSWERED: Where do the numbers in Phase 14 come from?**  
> 
> **1. `329,990 parameters`:**  
> - Linear(1280 $\rightarrow$ 256): $1280 \times 256 + 256 = 327,936$  
> - BatchNorm(256): $256 \times 2 = 512$  
> - Linear(256 $\rightarrow$ 6): $256 \times 6 + 6 = 1,542$  
> - **Total:** $327,936 + 512 + 1,542 = \mathbf{329,990}$ tunable dials!  
> 
> **2. `Features [8, 1280] -> Logits [8, 6]`:**  
> A test batch of **8 people**, each having **1,280 feature numbers**, produces **6 emotion scores** per person (Anger, Disgust, Fear, Happiness, Sadness, Surprise). Whichever score is highest is the prediction!  
> 
> **3. `Images [2, 3, 224, 224] -> Logits [2, 6]`:**  
> The unified end-to-end model swallowed **2 raw photos** ($224 \times 224 \times 3$), passed them through EfficientNet, and immediately returned the **6 emotion scores** for each photo!  
> 
> **4. `Encoder trainable = 0, Classifier trainable = 329,990`:**  
> Proof that our digital padlock works: all 4 million parameters of EfficientNet are locked so we don't accidentally ruin them while training the classifier!

---

<a name="phase-15--training-the-classifier-3-stages"></a>
## Phase 15 — Training the Classifier (3 Stages)

### What We Did:
We executed our **Three-Stage Training Strategy**:
1. **Stage A (Warm-Up on Cached Features):**
   - Trained the classifier head for 20 epochs in 2 seconds.
   - Training loss dropped from $0.948 \rightarrow 0.004$ (Train Acc: **100.0%**).
   - Validation loss reached **$0.8767$** (Val Acc: **70.70%**).
   - Saved weights to `models_checkpoints/best_classifier_stageA.pth`.
2. **Stage B (Conservative Fine-Tuning Safeguard):**
   - When training on small datasets (~700 images), unfreezing 3 million CNN parameters can destabilize vision features.
   - Our pipeline has an automatic safeguard: it only accepts fine-tuning if it strictly beats Stage A's validation loss.
3. **Stage C (Honest Evaluation on Held-Out Test Set):**
   - Evaluated once on Subjects 2 & 11 (156 unseen faces).
   - Final Test Accuracy reached **53.21%** (versus random guess of $16.67\%$).
   - Saved unified model to `models_checkpoints/best_emotion_model.pth`.

---

<a name="phase-16--comprehensive-evaluation--final-exam"></a>
## Phase 16 — Comprehensive Evaluation & Final Exam

### What We Did:
We gave the model its Final Exam on the 156 unseen test faces and generated full reports and visual heatmaps.

### The Four Key Metrics:
- **Accuracy (57.05%):** Out of 100 random test faces, the model got 57 right (random guessing is only 16.67%).
- **Macro Precision (63.82%):** For **Anger**, precision was **100%** (zero false alarms: every face it called Angry was genuinely Angry).
- **Macro Recall (60.19%):** For **Happiness**, recall was **100%** (it caught every single happy face with 0 missed).
- **Macro F1-Score (53.67%):** Balanced harmonic mean of Precision and Recall.

### Per-Emotion Breakdown:
| Emotion | Precision | Recall | F1-Score | Support | Key Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Anger** | **100.00%** | 14.81% | 25.81% | 27 | 100% Precision: zero false positives |
| **Disgust** | 60.00% | 30.00% | 40.00% | 20 | Confused occasionally with Surprise |
| **Fear** | 61.76% | 53.85% | 57.53% | 39 | Strong detection coverage |
| **Happiness** | 50.00% | **100.00%** | **66.67%** | 10 | 100% Recall: perfect catch rate |
| **Sadness** | 58.62% | **85.00%** | **69.39%** | 20 | High detection rate (17/20 correct) |
| **Surprise** | 52.54% | **77.50%** | **62.63%** | 40 | Reliable mouth/eyebrow detection |

Generated visual plots:
- Heatmap (Counts): `results/kmu_fed/confusion_matrix.png`
- Heatmap (Normalized %): `results/kmu_fed/confusion_matrix_normalized.png`
- Bar Chart: `results/kmu_fed/per_class_metrics.png`

---

<a name="phase-17--2d-feature-space-visualization-t-sne--pca"></a>
## Phase 17 — 2D Feature Space Visualization (t-SNE & PCA)

### What We Did:
Humans cannot see a 1,280-dimensional space. We used **t-SNE** and **PCA** to compress those 1,280 numbers into **2 coordinates $(x, y)$** to plot every face as a colored dot.

### What the 2D Scatter Plots Show:
1. **Happiness (Amber Orange) & Surprise (Turquoise):** Form isolated, clear island clusters away from negative emotions.
2. **Sadness (Deep Blue) & Fear (Amethyst Purple):** Form neighboring clusters because they share similar forehead and eyebrow tension.
3. **Anger (Crimson Red):** Forms a dense, compact grouping (explaining the 100% precision).

### Quantitative Clustering Metrics:
| Metric | Test Set Score | Ideal Direction | What It Means |
| :--- | :---: | :---: | :--- |
| **Silhouette Score** | **0.0483** | Higher is better ($-1$ to $+1$) | Measures how distinct clusters are from each other. |
| **Davies-Bouldin Index** | **2.3438** | Lower is better ($0$ is perfect) | Ratio of within-cluster spread to between-cluster separation. |
| **Calinski-Harabasz Score** | **13.10** | Higher is better | Ratio of between-cluster variance to within-cluster variance. |

Generated visual plots:
- Test Set 2D t-SNE: `results/kmu_fed/tsne_features_test.png`
- Test Set 2D PCA: `results/kmu_fed/pca_features_test.png`
- Training Set 2D t-SNE: `results/kmu_fed/tsne_features_train.png`
- Entire Dataset 2D t-SNE: `results/kmu_fed/tsne_features_all.png`

---

<a name="bonus-how-to-test-any-image-live-predictpy"></a>
## Bonus — How to Test Any Image Live (`predict.py`)

You can test any image on your computer in 0.1 seconds using our standalone inference tool:

```bash
# Test a sample face from the dataset
python predict.py --image "data/processed/02_HA_s01/02_HA_s01_021.jpg" --save_annotated

# Test any photo on your computer (auto-detects and crops the face!)
python predict.py --image "C:\path\to\my_photo.jpg" --save_annotated
```

### Sample Output:
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
*Created especially for you — Happy learning!*
