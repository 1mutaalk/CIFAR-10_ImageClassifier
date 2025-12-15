# CIFAR-10: Classical ML vs Deep Learning (PyTorch) 

This repo is my end‑to‑end machine learning project on the CIFAR‑10 image classification dataset. The main goal is to **compare classical machine learning models with a modern deep learning architecture** and show clearly where each approach shines, using a fair and well‑designed experimental setup.

---

## 1. Project Overview

### What this project does

- Trains **classical ML models** (SVM, k‑NN) on CIFAR‑10 using:
  - Hand‑crafted features (HOG).
  - Raw pixel features.
  - Deep features extracted from a ResNet‑18 network.
- Fine‑tunes a **deep learning model (ResNet‑18)** using PyTorch and transfer learning.
- Performs a **fair comparison** between:
  - Classical ML on hand‑crafted features.
  - Classical ML on deep features.
  - End‑to‑end deep learning.
- Provides a **comprehensive evaluation** with multiple metrics and visualizations.
- Is written as a **clean, reproducible notebook** (`notebooks/ML_project.ipynb`) that can be run on GPU (e.g., Google Colab).

### Why CIFAR‑10?

CIFAR‑10 is a classic benchmark dataset for image classification:

- 60,000 color images at 32×32 resolution.
- 10 object classes: `plane, car, bird, cat, deer, dog, frog, horse, ship, truck`.
- Challenging enough to show the limits of simple models, but small enough to train deep nets in a student / research environment.

---

## 2. Dataset and Splits

To keep the comparison fair and reproducible, I use a consistent splitting strategy:

- **Training set**: 50,000 images (official CIFAR‑10 training set).
- **Validation set**: 5,000 images (half of the official test set).
- **Final test set**: 5,000 images (other half of the official test set).

The same **val/test indices** are used for:

- Deep learning (ResNet‑18 training and validation).
- Classical ML models (HOG + SVM, k‑NN, and classical models on deep features).

A single `set_seed(42)` function seeds Python, NumPy, and PyTorch to keep runs as reproducible as possible.

---

## 3. Data Pipelines

### 3.1 Deep Learning Pipeline (PyTorch)

For the CNN / ResNet part, images are loaded as PyTorch tensors and passed through the following transforms:

- **Training transforms**:
  - `RandomCrop(32, padding=4)`
  - `RandomHorizontalFlip()`
  - `ToTensor()`
  - `Normalize(mean=(0.4914, 0.4822, 0.4465),
              std=(0.2470, 0.2435, 0.2616))`

- **Validation/Test transforms**:
  - `ToTensor()`
  - `Normalize(...)` with the same CIFAR‑10 mean and std.

These are wrapped into `DataLoader`s for:

- `train_loader`
- `val_loader`
- `final_test_loader`

### 3.2 Classical ML Pipeline (NumPy / scikit‑learn)

For classical models, I work with NumPy arrays derived from the same dataset:

- `train_ds_raw` and `test_ds_raw` keep the original CIFAR‑10 images in tensor form for:
  - **HOG feature extraction**.
  - **Raw pixel flattening**.
- Later, I also create **deep feature arrays** (`X_train_deep`, `X_test_deep`) using the ResNet feature extractor.

This way, classical ML and deep learning models always see consistent data and splits.

---

## 4. Models

### 4.1 Classical Model 1 – HOG + SVM

This is a more “traditional CV” pipeline:

1. **HOG feature extraction**
   - Convert images from tensor `(C, H, W)` to `(H, W, C)`.
   - Scale to `[0, 255]` and cast to `uint8`.
   - Use `skimage.feature.hog` with:
     - `orientations=8`
     - `pixels_per_cell=(8, 8)`
     - `cells_per_block=(2, 2)`
     - `channel_axis=-1`
   - This produces a HOG descriptor for each image that captures edges and gradient structure.

2. **Feature scaling**
   - Standardize HOG features with `StandardScaler`.

3. **SVM classifier**
   - `sklearn.svm.SVC` with RBF kernel.

4. **Hyperparameter tuning**
   - `GridSearchCV` on a subset of the training data (to keep it computationally reasonable):
     - `C ∈ {1, 10}`
     - `gamma ∈ {'scale'}`
   - 3‑fold cross‑validation.
   - Best hyperparameters are then used to train a final SVM on the full HOG feature set.

This model is a strong classical baseline: hand‑engineered image features + powerful kernel machine.

---

### 4.2 Classical Model 2 – k‑NN on Raw Pixels

This is the most direct classical baseline:

1. **Features**
   - Images from `train_ds_raw` are reshaped from `(N, 32, 32, 3)` to `(N, 3072)`.
   - Same for the test set (using the exact test indices as the deep learning model).

2. **Scaling**
   - `StandardScaler` on the flattened pixel vectors.

3. **k‑Nearest Neighbors**
   - `KNeighborsClassifier` from scikit‑learn.

4. **Hyperparameter tuning**
   - `GridSearchCV` on a subset:
     - `n_neighbors ∈ {3, 7}`
     - `weights ∈ {'distance'}`.
   - Once the best hyperparameters are found, the final k‑NN model is trained on all scaled training samples.

k‑NN is simple but gives a useful reference point: how far can we go with just raw pixels and no feature learning?

---

### 4.3 Deep Learning Model – ResNet‑18 (Transfer Learning)

For deep learning, I use a modern convolutional architecture with transfer learning:

1. **Base model**
   - `torchvision.models.resnet18(weights='DEFAULT')`
   - These weights are pretrained on ImageNet (1,000‑class large dataset).

2. **Modification for CIFAR‑10**
   - Replace the final fully connected layer:
     - Original: `fc: Linear(512 → 1000)`
     - New: `fc: Linear(512 → 10)` for CIFAR‑10.

3. **Training setup**
   - Device: GPU if available (`cuda`), otherwise CPU.
   - Loss: `nn.CrossEntropyLoss`.
   - Optimizer: `torch.optim.Adam` with learning rate `3e-4`.
   - Scheduler: `ReduceLROnPlateau` (monitors validation loss).
   - Epochs: up to 25 (with early stopping via best validation loss checkpoint).
   - Data:
     - Train on `train_loader` (with augmentation).
     - Validate on `val_loader` (no augmentation).

4. **Training loop**
   - For each epoch:
     - Train phase:
       - Forward pass → loss → backward → optimizer step.
     - Validation phase:
       - Compute validation loss and accuracy.
     - Scheduler step on validation loss.
     - Save the best model weights when validation loss improves.

ResNet‑18 acts as the main deep learning baseline and as a powerful feature extractor.

---

### 4.4 Hybrid Approach – Classical Models on Deep Features

This is where things get interesting: combining deep learning with classical ML.

1. **Feature extractor**
   - I define a `FeatureExtractor` module that:
     - Wraps the trained ResNet‑18.
     - Removes the final classification layer.
     - Outputs the penultimate (512‑dim) feature vector for each image.

2. **Deep feature extraction**
   - Pass all training and test images (from the same DataLoaders used during training) through this feature extractor.
   - Collect:
     - `X_train_deep` (shape ≈ `(50000, 512)`)
     - `X_test_deep` (shape ≈ `(5000, 512)`)
     - Corresponding labels `y_train_deep`, `y_test_deep`.

3. **Scaling**
   - Standardize deep features with `StandardScaler`.

4. **Classical models on deep features**
   - **SVM (RBF)**:
     - `kernel='rbf'`, `C=5` (optimized for these features).
   - **k‑NN**:
     - `n_neighbors=5`, `weights='distance'`.

These models show how much classical ML performance improves when we replace raw pixels / HOG with high‑quality deep features.

---

## 5. Evaluation

The project uses a **rich set of metrics and visualizations** for all main models.

### 5.1 Metrics

For each model (ResNet‑18, SVM on deep features, k‑NN on deep features, and optionally HOG + SVM / raw‑pixel k‑NN), I report:

- **Accuracy** on the final 5,000‑image test set.
- **Precision, recall, and F1‑score** per class via `classification_report`.
- **Macro and weighted F1‑scores** (to account for class imbalance).
- **Confusion matrix** as a heatmap (Seaborn).
- **Weighted multi‑class ROC‑AUC** (where probabilistic outputs are available).

This gives a much clearer picture than just a single accuracy number.

### 5.2 Fairness of comparison

To keep the comparison fair:

- All models use **the same train/val/test split**.
- Deep features are extracted from the **same trained ResNet‑18** that is evaluated end‑to‑end.
- Classical models on deep features have **no extra data**; they only get what the ResNet sees.

The final notebook cell produces a summary table that aligns all metrics side by side.

---

## 6. Typical Results (High‑Level)

Exact numbers can vary slightly by run, but typical behavior is:

- **ResNet‑18 (fine‑tuned)**  
  - Test accuracy ≈ **86%**.  
  - Strong, consistent performance across most classes.

- **SVM on deep features**  
  - Test accuracy ≈ **mid‑80%** (often very close to ResNet‑18).  
  - Shows that combining deep features with classical ML is very competitive.

- **k‑NN on deep features**  
  - Test accuracy ≈ **mid‑80%** as well.  
  - Demonstrates that even simple distance‑based models can perform well when given good embeddings.

- **HOG + SVM / k‑NN on raw pixels**  
  - Lower performance compared to deep feature‑based methods.  
  - Good as baselines to illustrate the value of feature learning.

The main takeaway: **feature learning from deep networks is crucial**, and classical models become much more powerful when they operate on those learned representations instead of raw pixels.

---
## 8. How to Run This Project

You can run this project either on **Google Colab** (easiest) or **locally** on your own machine. Everything is implemented in a single notebook: `notebooks/ML_project.ipynb`.

---

### 8.1 Running on Google Colab (Recommended)

1. Clone or download this repository.
2. Upload `notebooks/ML_project.ipynb` to your Google Drive.
3. In Google Drive, right‑click the notebook → **Open with → Google Colaboratory**.
4. In Colab, go to:  
   `Runtime → Change runtime type → Hardware accelerator → GPU → Save`.
5. Then run:  
   `Runtime → Run all`.
6. The notebook will automatically:
   - Download the CIFAR‑10 dataset.
   - Build train / validation / test splits.
   - Fine‑tune the ResNet‑18 model.
   - Extract deep features from the trained ResNet.
   - Train classical models (SVM, k‑NN) on those deep features.
   - Evaluate all models and print metrics + plots.

No manual edits are required; it is designed to run top‑to‑bottom.

---

### 8.2 Running Locally (Python)

1. **Clone the repo**

git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


2. **Create and activate a virtual environment** (optional but recommended)

python -m venv .venv

Windows:
.venv\Scripts\activate

macOS / Linux:
source .venv/bin/activate

3. **Install dependencies**

pip install torch torchvision torchaudio
pip install scikit-learn scikit-image matplotlib seaborn pandas

or, if present:
pip install -r requirements.txt


4. **Launch Jupyter and open the notebook**

5. In Jupyter, go to **Cell → Run All**.

If PyTorch detects a GPU, the first cell will show `Using device: cuda`; otherwise it will fall back to CPU (training will just be slower).

---

### 8.3 What You Get After Running

At the end of the notebook you will see:

- Final test accuracy and F1‑scores for:
  - Fine‑tuned ResNet‑18 (deep learning).
  - SVM on deep features.
  - k‑NN on deep features.
- Full classification reports (precision, recall, F1 for each class).
- Confusion matrix plots for visual error analysis.
- A summary table comparing **classical ML vs deep learning vs hybrid** approaches on the same CIFAR‑10 test set.

-------------

## 9. What this project achieves

- **Classical ML implementation**  
  - At least two classical models (HOG + SVM, k‑NN).  
  - Proper hyperparameter tuning with `GridSearchCV`.  
  - Careful preprocessing (HOG features, pixel standardization, deep feature scaling).

- **Deep learning implementation**  
  - Fine‑tuned ResNet‑18 with transfer learning, proper training loop, scheduler, and early stopping.  
  - Data augmentation and normalization tuned for CIFAR‑10.

- **Comparative analysis**  
  - Direct comparison between:
    - Classical models on handcrafted features.
    - Classical models on deep features.
    - End‑to‑end deep learning.
  - Same splits, same metrics, same test set.

- **Comprehensive evaluation**  
  - Accuracy, precision, recall, F1, confusion matrices, and ROC‑AUC.  
  - Per‑class and aggregate metrics.

- **Code quality**  
  - Clear sectioning in the notebook (imports, data, classical models, deep learning, feature extraction, evaluation).  
  - Reproducible seeds and consistent device handling.  
  - Comments and variable names that make the logic easy to follow.

---

## 10. Authors

- **Primary author:** *Muhammad Mutaal Khan* – designed, implemented, and documented the pipeline.
- **Project collaborators:**  
  - *Saif Ullah Farooqi* – contributed to model tuning and evaluation.  

---

## 11. Closing Thoughts

This project is my attempt to build a **clean, serious comparison** between classical ML and deep learning on a real vision task, while keeping the code approachable for students and practitioners.  

If you find this useful, feel free to open issues, suggest improvements, or use parts of the notebook for your own experiments. ⭐ are always appreciated!


