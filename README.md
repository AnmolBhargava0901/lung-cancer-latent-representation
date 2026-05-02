# 🧬 Lung Cancer Gene Expression Analysis — Autoencoder-Based Pipeline

A machine learning framework for lung cancer gene expression analysis using a sequential pipeline that transforms raw gene expression data into therapy-oriented insights. The system merges supervised and unsupervised learning to ensure efficient feature extraction and reliable result validation.

---

## 📋 Table of Contents

- [System Design](#-system-design)
- [Datasets](#-datasets)
- [Autoencoder Architecture](#-autoencoder-semantic-embedding-architecture)
- [Classification Model](#-classification-based-feature-validation)
- [Gene Prioritization & Therapy Categorization](#-gene-prioritization--therapy-categorization)
- [Results & Visualizations](#-results--visualizations)
- [Tools & Technologies](#-tools--technologies)
- [Repository Structure](#️-repository-structure)

---

## 🔬 System Design

The framework utilizes a **modular, step-by-step sequential pipeline** designed to convert raw gene expression data into meaningful, treatment-oriented insights. Each stage of the pipeline contributes to reducing data complexity while preserving essential biological information.

### Pipeline Stages

```
1. Data Acquisition and Preprocessing
          │
          ▼
2. Latent Representation Learning using Autoencoder
          │
          ▼
3. Classification-Based Feature Validation
          │
          ▼
4. Gene Importance Ranking and Projection
          │
          ▼
5. Therapy-Oriented Gene Categorization
```

The entire workflow maintains a balance between **predictive accuracy** and **biological relevance**, overcoming major shortcomings of conventional methods. The method is designed to be adaptive and extensible for different gene expression datasets without major modifications to the framework. The modular structure allows for improvement or substitution of specific parts without affecting the overall process.

---

## 📦 Datasets

High-quality and biologically relevant datasets were selected from the **National Centre for Biotechnology Information Gene Expression Omnibus (NCBI GEO)** to ensure diversity in population samples and robustness in model evaluation.

| Dataset | Region | Samples | Composition |
|---|---|---|---|
| [GSE30118](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30118) | North-East India | 7 | 5 tumour, 2 pooled normal |
| [GSE10072](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10072) | United States | 107 | 58 tumour, 49 non-tumour |

The datasets represent **different geographical populations**, enabling the model to capture variation in gene expression patterns across diverse cohorts.

### Data Preprocessing Steps

- **Normalization** — Adjusted gene expression values across samples to reduce technical variations
- **Noise Correction** — Removal of genes showing low variance
- **Missing Values** — Handled using appropriate statistical methods

---

## 🤖 Autoencoder Semantic Embedding Architecture

The framework constructs a representation for the gene expression data using an **autoencoder**. The encoder layer maps high-dimensional data (~19,700 genes) into a low-dimensional latent space vector.

### Mathematical Formulation

```
Latent Representation:    z = f_θ(x)
Reconstruction:           x̂ = g_ø(z)
Objective (minimize):     ||x − x̂||²
```

The autoencoder is trained to learn a compact and meaningful representation of high-dimensional gene expression data by **minimizing the reconstruction error** between the input and its reconstructed output.

- The **encoder** `f_θ(x)` maps input data `x` into a lower-dimensional latent space `z`, where `θ` represents the learnable parameters
- The **decoder** `g_ø(z)` reconstructs the input data `x̂` from the latent representation, parameterized by `ø`
- The use of **non-linear activation functions** allows the model to capture complex relationships between genes that are not possible with linear methods

### Model Architecture

| Layer | Dimensions |
|---|---|
| Input | ~19,700 genes |
| Encoder Layer 1 | 1024 |
| Encoder Layer 2 | 256 |
| Latent Space | **64** |
| Activation (hidden) | ReLU |
| Activation (latent) | tanh |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |

---

## 🧠 Classification-Based Feature Validation

The learned **64-dimensional latent representations** are used as input to a minimalistic classifier to differentiate cancerous instances from normal instances.

### Mathematical Formulation

```
Output:    ŷ = σ(Wz + b)
Loss:      L_BCE = BCE(y, ŷ)
```

Where `W` is the weight matrix, `b` is the bias, and `σ` is the sigmoid activation function. Binary Cross-Entropy (BCE) is minimized during training so outcomes align more closely with ground truth.

### Classifier Architecture

| Component | Details |
|---|---|
| Input | 64-dimensional latent features |
| Hidden Layer 1 | 32 neurons |
| Hidden Layer 2 | 16 neurons |
| Dropout | 0.3 |
| Output Activation | Sigmoid (binary classification) |
| Loss Function | Binary Cross-Entropy |

### Training Strategy

- **80–20 train-test split** with stratification
- **5-fold cross-validation** for robustness analysis

---

## 🧬 Gene Prioritization & Therapy Categorization

The framework extends beyond classification by identifying and interpreting genes that significantly contribute to lung carcinoma progression.

### 4.1 Gene Importance Ranking

Latent features are projected back into the original gene space. Gene importance is quantified based on each gene's contribution to the learned latent representation:

```
Importance(gᵢ) = Σⱼ |Wᵢⱼ|
```

Where `Wᵢⱼ` denotes the weight connecting the `i`-th gene to the `j`-th neuron in the encoder layer. Taking the absolute value ensures that **both positive and negative contributions** are considered equally significant.

### 4.2 Direction of Regulation — logFC Analysis

Log Fold Change (logFC) analysis compares average gene expression levels between normal and cancerous samples:

```
logFC = log₂ ( Cancer Expression / Normal Expression )
```

- **Positive logFC** → Gene is **upregulated** in cancer
- **Negative logFC** → Gene is **downregulated** in cancer

### 4.3 Upregulated vs Downregulated Genes

| Category | logFC | Biological Role | Therapeutic Strategy |
|---|---|---|---|
| **Upregulated** | Positive | Overexpressed in cancer; may act as oncogenes | Gene silencing — siRNA / CRISPR knockout |
| **Downregulated** | Negative | Expressed at lower levels; may act as tumour suppressors | Gene activation / replacement therapy |

This classification bridges the gap between computational analysis and **precision medicine** by enabling actionable insights for targeted treatment approaches.

---

## 📊 Results & Visualizations

### Latent Space — PCA of Autoencoder Features<img width="4164" height="1770" alt="dataset_comparison" src="https://github.com/user-attachments/assets/8a694310-d4cb-4987-b200-d4c9ec09883b" />


![Latent Space PCA](1_latent_space_pca.png)

> PCA visualization of the 64-dimensional latent space learned by the autoencoder. Training and validation loss curves were examined to observe model convergence and identify possible problems like overfitting or underfitting.

---

### Confusion Matrix

![Confusion Matrix](2_confusion_matrix.png)

> Confusion matrix analysis providing a comprehensive insight into classification results — encompassing true positives, false positives, true negatives, and false negatives. This assists in discovering any bias or inequity in the model's predictions.

---

### ROC Curve

![ROC Curve](3_roc_curve.png)

> ROC curve evaluating the classifier's ability to discriminate between cancerous and normal tissue samples across classification thresholds.

---

### Performance Metrics

![Performance Metrics](4_performance_metrics.png<img width="2964" height="1769" alt="4_performance_metrics" src="https://github.com/user-attachments/assets/b311c439-f3cf-46ba-b13e-3ad9203b663e" />
)

> Performance comparison across training and validation, used alongside 5-fold cross-validation to assess robustness of the learned latent representations.

---

### Gene Feature Importance

![Gene Importance](gene_importance.![Uploading 4_performance_metrics.png…]()
<img width="3562" height="2364" alt="gene_importance" src="https://github.com/user-attachments/assets/ad975c32-dd2f-4edc-9a9b-97451d5a395a" />
png)

> Top genes ranked by their contribution scores derived from encoder input layer weights. Genes with higher influence on the latent space receive higher importance scores.

---


### Dataset Comparison

![Dataset Comparison](<img width="4164" height="1770" alt="dataset_comparison" src="https://github.com/user-attachments/assets/2b784c0b-bd18-476d-826a-c6c171b318dc" /> )

> Side-by-side comparison of the USA (GSE10072) and India (GSE30118) datasets, representing different geographical populations used for training and external validation.

---

## 🛠️ Tools & Technologies

### Programming Environment

| Tool | Purpose |
|---|---|
| **Python 3.x** | Core programming language for scientific computing and machine learning |
| **Jupyter Notebook** | Interactive coding, visualization, and experimentation |
| **Visual Studio Code** | Development environment with debugging support |

### Data Handling & Preprocessing

| Library | Purpose |
|---|---|
| **Pandas** | Handling structured gene expression datasets — cleaning, transformation, aggregation |
| **NumPy** | Numerical computations and efficient handling of large multidimensional arrays |

### Machine Learning & Deep Learning

| Library | Purpose |
|---|---|
| **Keras (TensorFlow backend)** | Building and training the autoencoder and classification model |
| **Scikit-learn** | Preprocessing, model evaluation, cross-validation, confusion matrix generation |

### Visualization

| Library | Purpose |
|---|---|
| **Matplotlib** | Plotting training loss curves, performance comparisons, gene importance rankings |

---

## 🗂️ Repository Structure

```
lung-cancer-genomic-ai/
│
├── 📄 dual_dataset_pipeline.py          # Full dual-dataset analysis (USA + India)
├── 📄 final_publication_pipeline.py     # Publication-ready pipeline
│
├── 📊 usa_lung_cancer_ml_ready.csv      # Processed USA dataset (GSE10072)
├── 📊 india_lung_cancer_ml_ready.csv    # Processed India dataset (GSE30118)
├── 📊 top_50_genes.csv                  # Top biomarker genes with importance scores
│
├── 📝 FINAL_SUMMARY.md                  # Detailed results and interpretation
├── 📝 PUBLICATION_GUIDELINES.txt        # Manuscript reporting guidelines
│
├── 🖼️ 1_latent_space_pca.png            # PCA of autoencoder latent space
├── 🖼️ 2_confusion_matrix.png            # Classification confusion matrix
├── 🖼️ 3_roc_curve.png                   # ROC curve
├── 🖼️ 4_performance_metrics.png         # Train / Validation performance comparison
├── 🖼️ gene_importance.png               # Top gene feature importances
└── 🖼️ dataset_comparison.png            # USA vs India dataset overview
```

---

## 📄 License

This project is intended for academic and research use. Gene expression datasets are publicly available through NCBI GEO under their respective accession terms. Please cite the original data sources when publishing results derived from this pipeline.

---

<div align="center">

**Built with** `Keras` · `TensorFlow` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib`

*Datasets: GSE10072 (USA) + GSE30118 (North-East India)*

</div>
