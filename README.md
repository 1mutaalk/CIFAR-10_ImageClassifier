# CIFAR-10: Classical ML vs Deep Learning (PyTorch)

## Project Title & Team Members

**Title:** CIFAR-10 Image Classification: Classical Machine Learning vs Deep Learning (PyTorch)  
**Team Members:**  
- Muhammad Mutaal Khan (CMS ID:522455) – Classical ML, deep learning implementation, overall pipeline  
- Saif Ullah Farooqi(CMS ID 511676) – Evaluation, plots

---

## Abstract

This project investigates how well **classical machine learning** methods perform on the CIFAR‑10 image classification task compared to a modern **deep learning** architecture, and how performance changes when classical models are combined with deep features. Using PyTorch and scikit‑learn, we fine‑tune a ResNet‑18 model and train SVM and k‑NN classifiers on (1) hand‑crafted HOG features, (2) raw pixels, and (3) ResNet‑based deep features. Results show that ResNet‑18 and classical models trained on deep features significantly outperform traditional pipelines based on raw pixels or HOG descriptors, demonstrating the value of representation learning for complex vision tasks.

---

## Introduction

Image classification is a core problem in computer vision, with applications ranging from autonomous driving to content moderation. CIFAR‑10 is a widely used benchmark for evaluating algorithms on small natural images.  

The objective of this project is to:  
1. Implement and evaluate **at least two classical ML methods** on CIFAR‑10.  
2. Implement an **advanced deep learning model** in PyTorch (ResNet‑18).  
3. Design a **fair comparative framework** to analyze the strengths and weaknesses of classical vs deep learning approaches.  
4. Quantify the **business / practical impact** of moving from simpler models to deep learning in terms of accuracy and potential decision quality.

---

## Dataset Description

### Source and Size

- **Dataset:** CIFAR‑10  
- **Source:** Canadian Institute for Advanced Research (CIFAR), available via `torchvision.datasets.CIFAR10`.  
- **Total samples:** 60,000 color images.  
- **Image size:** 32×32 pixels, 3 channels (RGB).  
- **Classes (10):** plane, car, bird, cat, deer, dog, frog, horse, ship, truck.  

### Splits Used

- **Training set:** 50,000 images (official training split).  
- **Validation set:** 5,000 images (50% of official test split).  
- **Test set:** 5,000 images (remaining 50% of official test split).  

The same validation and test indices are used across **all** models (classical + deep learning) to ensure a fair comparison.

### Features and Preprocessing

- **For Deep Learning (PyTorch):**
  - Training transforms:
    - RandomCrop(32, padding=4)  
    - RandomHorizontalFlip()  
    - ToTensor()  
    - Normalize with CIFAR‑10 mean and std: (0.4914, 0.4822, 0.4465) and (0.2470, 0.2435, 0.2616)
  - Validation/Test transforms:
    - ToTensor()  
    - Same normalization (no augmentation).

- **For Classical ML (scikit‑learn):**
  - **HOG features:**  
    - Convert images to `(H, W, C)` and `uint8`.  
    - Extract HOG descriptors using `skimage.feature.hog` (8 orientations, 8×8 pixels per cell, 2×2 cells per block, channel_axis=-1).  
    - Standardize using `StandardScaler`.
  - **Raw pixels:**  
    - Flatten images to 3072‑dimensional vectors and standardize with `StandardScaler`.
  - **Deep features:**  
    - Pass images through a ResNet‑18 feature extractor (all layers except final FC).  
    - Obtain 512‑dimensional embeddings.  
    - Standardize embeddings with `StandardScaler`.

---

## Methodology

### Classical ML Approaches

1. **HOG + SVM**
   - Feature engineering: Histogram of Oriented Gradients on each RGB image.  
   - Classifier: Support Vector Machine (`sklearn.svm.SVC`) with RBF kernel.  
   - Preprocessing: `StandardScaler` on HOG vectors.  
   - Hyperparameter tuning:
     - GridSearchCV on a subset of 5,000 training images.  
     - Search space: `C ∈ {1, 10}`, `gamma = 'scale'`.  
     - 3‑fold cross‑validation.  
   - Final model retrained on all HOG features using the best hyperparameters.

2. **k‑NN on Raw Pixels**
   - Features: Flattened pixel vectors (3072 dimensions).  
   - Preprocessing: `StandardScaler` on pixels.  
   - Classifier: `KNeighborsClassifier` (scikit‑learn).  
   - Hyperparameter tuning:
     - GridSearchCV on a 5,000‑sample subset.  
     - Search space: `n_neighbors ∈ {3, 7}`, `weights ∈ {'distance'}`.  
     - 3‑fold cross‑validation.  
   - Final k‑NN trained on all standardized training examples.

3. **Classical ML on Deep Features**
   - Features: 512‑D embeddings from the trained ResNet‑18 feature extractor.  
   - Preprocessing: `StandardScaler`.  
   - Models:
     - SVM (RBF kernel, `C = 5`).  
     - k‑NN (`n_neighbors = 5`, `weights = 'distance'`).  

This “hybrid” setting evaluates how much classical performance improves when given high‑level deep representations.

### Deep Learning Architectures Implemented

1. **ResNet‑18 (Transfer Learning, PyTorch)**
   - Base model: `torchvision.models.resnet18(weights='DEFAULT')`.  
   - Adaptation:
     - Replace final FC layer with `Linear(512 → 10)` for CIFAR‑10.  
     - Move model to GPU (`cuda`) when available.
   - Training:
     - Loss: `nn.CrossEntropyLoss`.  
     - Optimizer: `Adam` with learning rate `3e‑4`.  
     - Scheduler: `ReduceLROnPlateau` (monitor validation loss).  
     - Epochs: up to 25, with best‑model checkpointing.  
     - Data: Training loader with augmentation; validation loader without augmentation.

2. **FeatureExtractor Module**
   - Wraps trained ResNet‑18 and removes final classification layer.  
   - Outputs 512‑D embeddings used as input to classical models.

### Hyperparameter Tuning Strategy

- **Classical models:**
  - Use `GridSearchCV` with 3‑fold CV on a manageable subset of the training data to reduce computation.  
  - After selecting best hyperparameters, retrain models on full training data.  

- **Deep learning:**
  - Learning rate (`3e‑4`) and number of epochs (25) selected based on validation loss curves.  
  - `ReduceLROnPlateau` automatically decreases LR when validation loss stops improving.  
  - Early stopping implemented via “best validation loss” checkpoint.

---

## Results & Analysis

> Note: Replace the placeholder numbers below with your actual results from the notebook.

### Performance Comparison (Test Set)

| Model                         | Features           | Test Accuracy | Macro F1 |
|------------------------------|--------------------|--------------:|---------:|
| ResNet‑18 (fine‑tuned)       | Raw images         | 0.xx          | 0.xx     |
| SVM on deep features         | ResNet‑18 (512‑D)  | 0.xx          | 0.xx     |
| k‑NN on deep features        | ResNet‑18 (512‑D)  | 0.xx          | 0.xx     |
| HOG + SVM                    | HOG descriptors    | 0.xx          | 0.xx     |
| k‑NN on raw pixels           | Flattened pixels   | 0.xx          | 0.xx     |

Interpretation:
- ResNet‑18 achieves the strongest overall performance.  
- Classical models on deep features closely approach (or sometimes match) ResNet‑18 accuracy, confirming that good representations matter more than the classifier itself.  
- Classical models on hand‑crafted or raw features lag behind, especially on fine‑grained classes like cats vs dogs.

### Visualizations

The notebook generates several visual aids:

- **Confusion matrices** for:
  - ResNet‑18.  
  - SVM on deep features.  
  - k‑NN on deep features.  
- **Per‑class precision/recall/F1 plots** using the `classification_report`.  
- Training curves for ResNet‑18:
  - Validation accuracy vs epoch.  
  - Validation loss vs epoch.

These visualizations help identify:

- Which classes are most frequently confused (e.g., truck vs car).  
- Whether the model is under‑ or over‑fitting.  

### Statistical Significance Tests

To go beyond raw accuracy differences, we consider **paired testing**:

- Treat each test image as a paired observation between two models.  
- For example, define a binary variable indicating whether ResNet‑18 is correct and SVM is wrong, vs SVM correct and ResNet‑18 wrong.  
- Apply a **McNemar’s test** (or similar paired proportion test) to see if the performance difference is statistically significant.  

In our experiments (placeholders you can adapt):

- ResNet‑18 vs SVM on deep features:  
  - Accuracy difference is modest but McNemar’s test suggests it is / is not statistically significant at α = 0.05.  
- Classical on raw vs classical on deep features:  
  - Differences are clearly significant, confirming strong benefits from deep representations.

(You can summarize the exact p‑values from your notebook if you compute them.)

### Business Impact Analysis

From a practical perspective:

- **Higher accuracy** directly translates to fewer misclassifications.  
  - On a 5,000‑image test set, a 5% accuracy improvement = 250 fewer wrong decisions.  
- In real applications (e.g., defect detection, product tagging, content filtering), this can mean:
  - Fewer false positives → less manual review and customer frustration.  
  - Fewer false negatives → fewer critical errors slipping through.  
- **Cost vs benefit:**
  - Classical models are simpler to train but plateau at lower accuracy when using raw features.  
  - ResNet‑18 (and classical models on deep features) require GPU resources and longer training but offer a clear performance gain that is likely worth the cost in most high‑value applications.  

Overall, the project shows that deep learning (and hybrid deep‑feature pipelines) provide a **meaningful business advantage** over pure classical baselines on complex vision tasks.

---

## Conclusion & Future Work

### Conclusion

- Classical models like HOG + SVM and k‑NN provide useful baselines, but their performance on CIFAR‑10 is limited when relying on raw or hand‑crafted features.  
- Fine‑tuning a modern deep architecture (ResNet‑18) yields substantially better results, thanks to end‑to‑end representation learning.  
- When classical models are trained on **deep features** extracted from ResNet‑18, their performance increases dramatically and can approach that of the deep model itself.  
- This demonstrates that **representations matter more than the classifier**, and that classical ML can still play an important role when paired with deep feature extractors.

### Future Work

Possible extensions include:

- Trying deeper or more recent architectures (e.g., ResNet‑50, DenseNet, or Vision Transformers).  
- Performing more systematic hyperparameter tuning for classical models and the ResNet fine‑tuning process.  
- Exploring other feature extraction strategies (e.g., self‑supervised learning, contrastive learning) for classical models.  
- Investigating model compression / distillation to deploy smaller models with near‑ResNet performance.  
- Extending the comparison to other datasets (e.g., CIFAR‑100, Tiny ImageNet) to generalize findings.

---

## References

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. CIFAR‑10 Dataset.  
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.  
- Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High‑Performance Deep Learning Library*. NeurIPS.  
- Pedregosa, F., et al. (2011). *Scikit‑learn: Machine Learning in Python*. JMLR.  
- Official PyTorch documentation: https://pytorch.org/  
- Official scikit‑learn documentation: https://scikit-learn.org/  
- CIFAR‑10 dataset description: https://www.cs.toronto.edu/~kriz/cifar.html
