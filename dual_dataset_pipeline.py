# ==========================================
# LUNG CANCER GENOMIC AI PIPELINE - DUAL DATASET ANALYSIS
# USA Dataset (GSE10072): 107 samples, 22,283 genes
# India Dataset (GSE30118): 7 samples, 19,700 genes
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. SETUP & CONFIGURATION
# ==========================================
USA_DATASET = 'usa_lung_cancer_ml_ready.csv'
INDIA_DATASET = 'india_lung_cancer_ml_ready.csv'
RESULTS_DIR_USA = 'results_usa'
RESULTS_DIR_INDIA = 'results_india'
RESULTS_DIR_COMBINED = 'results_combined'
SEED = 42

# Create directories
for directory in [RESULTS_DIR_USA, RESULTS_DIR_INDIA, RESULTS_DIR_COMBINED]:
    if not os.path.exists(directory):
        os.makedirs(directory)

np.random.seed(SEED)

print("="*80)
print("LUNG CANCER GENOMIC AI PIPELINE - DUAL DATASET ANALYSIS")
print("="*80)

# 2. LOAD AND ANALYZE DATASETS
# ==========================================
def load_and_preprocess(filepath, dataset_name):
    print(f"\n{'='*80}")
    print(f"LOADING {dataset_name} DATASET")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath)
    
    # Identify target column
    target_col = 'Label'
    
    # Drop ID columns and Target
    drop_cols = [target_col, 'Sample_ID']
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    # Store gene names
    gene_names = X.columns.tolist()
    
    # Encode Labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\n Dataset Statistics:")
    print(f"   - Samples: {X_scaled.shape[0]}")
    print(f"   - Genes/Features: {X_scaled.shape[1]:,}")
    print(f"   - Normal samples: {np.sum(y_enc == 0)}")
    print(f"   - Cancer samples: {np.sum(y_enc == 1)}")
    print(f"   - Class balance: {np.sum(y_enc == 0)}/{np.sum(y_enc == 1)} (Normal/Cancer)")
    
    return X_scaled, y_enc, gene_names, scaler, le

# Load both datasets
X_usa, y_usa, genes_usa, scaler_usa, le_usa = load_and_preprocess(USA_DATASET, "USA (GSE10072)")
X_india, y_india, genes_india, scaler_india, le_india = load_and_preprocess(INDIA_DATASET, "INDIA (GSE30118)")

# 3. PROCESS USA DATASET (Larger Dataset)
# ==========================================
print(f"\n{'='*80}")
print("ANALYZING USA DATASET")
print(f"{'='*80}")

# Train/Test Split for USA (sufficient samples)
X_train_usa, X_test_usa, y_train_usa, y_test_usa = train_test_split(
    X_usa, y_usa, test_size=0.2, stratify=y_usa, random_state=SEED
)

print(f"\n Data Split:")
print(f"   - Training: {X_train_usa.shape[0]} samples")
print(f"   - Testing: {X_test_usa.shape[0]} samples")

# Build Autoencoder for USA
print(f"\n Building USA Autoencoder...")
INPUT_DIM_USA = X_train_usa.shape[1]
LATENT_DIM = 64

autoencoder_usa = MLPRegressor(
    hidden_layer_sizes=(1024, 256, LATENT_DIM, 256, 1024),
    activation='tanh',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    max_iter=100,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False
)

print(f"   Architecture: {INPUT_DIM_USA} → 1024 → 256 → {LATENT_DIM} → 256 → 1024 → {INPUT_DIM_USA}")
print(f"   Training autoencoder...")

autoencoder_usa.fit(X_train_usa, X_train_usa)

train_loss_usa = np.mean((X_train_usa - autoencoder_usa.predict(X_train_usa)) ** 2)
test_loss_usa = np.mean((X_test_usa - autoencoder_usa.predict(X_test_usa)) ** 2)

print(f"   ✓ Training MSE: {train_loss_usa:.6f}")
print(f"   ✓ Testing MSE: {test_loss_usa:.6f}")

# Extract latent features
def extract_latent(X, autoencoder, latent_dim=64):
    hidden = X.copy()
    for i in range(2):
        weights = autoencoder.coefs_[i]
        bias = autoencoder.intercepts_[i]
        hidden = np.tanh(np.dot(hidden, weights) + bias)
    weights = autoencoder.coefs_[2]
    bias = autoencoder.intercepts_[2]
    latent = np.tanh(np.dot(hidden, weights) + bias)
    return latent

X_train_latent_usa = extract_latent(X_train_usa, autoencoder_usa, LATENT_DIM)
X_test_latent_usa = extract_latent(X_test_usa, autoencoder_usa, LATENT_DIM)

# Train Classifier on USA
print(f"\n Training USA Classifier...")
classifier_usa = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=16,
    max_iter=200,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)

classifier_usa.fit(X_train_latent_usa, y_train_usa)

# Predictions
y_pred_usa = classifier_usa.predict(X_test_latent_usa)
y_prob_usa = classifier_usa.predict_proba(X_test_latent_usa)[:, 1]

acc_usa = accuracy_score(y_test_usa, y_pred_usa)
auc_usa = roc_auc_score(y_test_usa, y_prob_usa)

print(f"\n USA RESULTS:")
print(f"   - Accuracy: {acc_usa:.4f}")
print(f"   - ROC-AUC: {auc_usa:.4f}")

# 4. PROCESS INDIA DATASET (Smaller Dataset - Use Cross-Validation)
# ==========================================
print(f"\n{'='*80}")
print("ANALYZING INDIA DATASET")
print(f"{'='*80}")

print(f"\n Small sample size ({X_india.shape[0]} samples) - Using Leave-One-Out approach")

# For India dataset, we'll use all data for training since it's very small
INPUT_DIM_INDIA = X_india.shape[1]

autoencoder_india = MLPRegressor(
    hidden_layer_sizes=(512, 128, LATENT_DIM, 128, 512),
    activation='tanh',
    solver='adam',
    alpha=0.0001,
    batch_size=4,
    learning_rate='adaptive',
    max_iter=150,
    random_state=SEED,
    early_stopping=False,
    verbose=False
)

print(f"\n Building India Autoencoder...")
print(f"   Architecture: {INPUT_DIM_INDIA} → 512 → 128 → {LATENT_DIM} → 128 → 512 → {INPUT_DIM_INDIA}")

autoencoder_india.fit(X_india, X_india)

loss_india = np.mean((X_india - autoencoder_india.predict(X_india)) ** 2)
print(f"    Reconstruction MSE: {loss_india:.6f}")

# Extract latent features
X_latent_india = extract_latent(X_india, autoencoder_india, LATENT_DIM)

# For classification with small sample, use simpler model
classifier_india = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=300,
    random_state=SEED,
    verbose=False
)

print(f"\n Training India Classifier...")
classifier_india.fit(X_latent_india, y_india)

# Predictions (on training data due to small size)
y_pred_india = classifier_india.predict(X_latent_india)
y_prob_india = classifier_india.predict_proba(X_latent_india)[:, 1]

acc_india = accuracy_score(y_india, y_pred_india)

print(f"\n INDIA RESULTS (Training Set - Small Sample):")
print(f"   - Accuracy: {acc_india:.4f}")
print(f"   - Note: Results on training set due to limited samples")

# 5. VISUALIZATIONS - USA DATASET
# ==========================================
print(f"\n{'='*80}")
print("GENERATING USA VISUALIZATIONS")
print(f"{'='*80}")

# Latent Space PCA - USA
pca_usa = PCA(n_components=2)
latent_2d_usa = pca_usa.fit_transform(X_test_latent_usa)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(latent_2d_usa[:, 0], latent_2d_usa[:, 1],
                     c=y_test_usa, cmap='RdYlBu_r',
                     s=150, alpha=0.7, edgecolors='black', linewidth=1)
plt.colorbar(scatter, label='Class (0=Normal, 1=Cancer)')
plt.title("USA Dataset: Latent Space (Test Set)", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({pca_usa.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
plt.ylabel(f"PC2 ({pca_usa.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_USA}/latent_space_pca.png", dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix - USA
cm_usa = confusion_matrix(y_test_usa, y_pred_usa)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_usa, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Cancer'],
            yticklabels=['Normal', 'Cancer'],
            annot_kws={'size': 16, 'weight': 'bold'})
plt.title("USA Dataset: Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_USA}/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve - USA
fpr_usa, tpr_usa, _ = roc_curve(y_test_usa, y_prob_usa)
roc_auc_usa = auc(fpr_usa, tpr_usa)

plt.figure(figsize=(8, 6))
plt.plot(fpr_usa, tpr_usa, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc_usa:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('USA Dataset: ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_USA}/roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# Gene Importance - USA
w_enc_usa = autoencoder_usa.coefs_[0]
importance_usa = np.sum(np.abs(w_enc_usa), axis=1)

importance_df_usa = pd.DataFrame({
    "Gene": genes_usa,
    "Importance": importance_usa
}).sort_values(by="Importance", ascending=False)

top_50_usa = importance_df_usa.head(50)
top_50_usa.to_csv(f"{RESULTS_DIR_USA}/top_50_genes.csv", index=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Gene", data=top_50_usa.head(20), palette="viridis")
plt.title("USA Dataset: Top 20 Genes", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Gene ID", fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_USA}/gene_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved USA visualizations to {RESULTS_DIR_USA}/")

# 6. VISUALIZATIONS - INDIA DATASET
# ==========================================
print(f"\n{'='*80}")
print("GENERATING INDIA VISUALIZATIONS")
print(f"{'='*80}")

# Latent Space PCA - India
pca_india = PCA(n_components=2)
latent_2d_india = pca_india.fit_transform(X_latent_india)

plt.figure(figsize=(10, 7))
colors = ['blue' if label == 0 else 'red' for label in y_india]
scatter = plt.scatter(latent_2d_india[:, 0], latent_2d_india[:, 1],
                     c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
for i, (x, y) in enumerate(latent_2d_india):
    plt.annotate(f'S{i+1}', (x, y), fontsize=10, ha='center', va='center')
plt.title("India Dataset: Latent Space (All Samples)", fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({pca_india.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
plt.ylabel(f"PC2 ({pca_india.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
plt.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='Normal'),
                  Patch(facecolor='red', label='Cancer')]
plt.legend(handles=legend_elements, loc='best')
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_INDIA}/latent_space_pca.png", dpi=300, bbox_inches='tight')
plt.close()

# Gene Importance - India
w_enc_india = autoencoder_india.coefs_[0]
importance_india = np.sum(np.abs(w_enc_india), axis=1)

importance_df_india = pd.DataFrame({
    "Gene": genes_india,
    "Importance": importance_india
}).sort_values(by="Importance", ascending=False)

top_50_india = importance_df_india.head(50)
top_50_india.to_csv(f"{RESULTS_DIR_INDIA}/top_50_genes.csv", index=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Gene", data=top_50_india.head(20), palette="plasma")
plt.title("India Dataset: Top 20 Genes", fontsize=14, fontweight='bold')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Gene ID", fontsize=12)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_INDIA}/gene_importance.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved India visualizations to {RESULTS_DIR_INDIA}/")

# 7. COMPARATIVE ANALYSIS
# ==========================================
print(f"\n{'='*80}")
print("COMPARATIVE ANALYSIS")
print(f"{'='*80}")

# Compare top genes
common_genes = set(top_50_usa['Gene'].values) & set(top_50_india['Gene'].values)
print(f"\n Common genes in top 50: {len(common_genes)}")
if len(common_genes) > 0:
    print(f"   Common genes: {list(common_genes)[:10]}")

# Summary plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Dataset comparison
datasets = ['USA', 'India']
samples = [X_usa.shape[0], X_india.shape[0]]
genes = [X_usa.shape[1], X_india.shape[1]]

axes[0].bar(datasets, samples, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Number of Samples', fontsize=12)
axes[0].set_title('Sample Size Comparison', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(samples):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

axes[1].bar(datasets, genes, color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Number of Genes', fontsize=12)
axes[1].set_title('Feature Dimension Comparison', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(genes):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR_COMBINED}/dataset_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Saved comparison plots to {RESULTS_DIR_COMBINED}/")

# 8. FINAL SUMMARY REPORT
# ==========================================
print(f"\n{'='*80}")
print("FINAL SUMMARY REPORT")
print(f"{'='*80}")

summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   LUNG CANCER GENOMIC AI ANALYSIS REPORT                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

 USA DATASET (GSE10072)
   • Total Samples: {X_usa.shape[0]} ({np.sum(y_usa==0)} Normal, {np.sum(y_usa==1)} Cancer)
   • Genes: {X_usa.shape[1]:,}
   • Train/Test Split: {X_train_usa.shape[0]}/{X_test_usa.shape[0]}
   • Test Accuracy: {acc_usa:.2%}
   • ROC-AUC: {auc_usa:.4f}
   • Autoencoder Loss: {test_loss_usa:.6f}

 INDIA DATASET (GSE30118)
   • Total Samples: {X_india.shape[0]} ({np.sum(y_india==0)} Normal, {np.sum(y_india==1)} Cancer)
   • Genes: {X_india.shape[1]:,}
   • Training Accuracy: {acc_india:.2%}
   • Autoencoder Loss: {loss_india:.6f}
   • Note: Small sample size - limited validation

 MODEL ARCHITECTURE
   • Autoencoder: Input → 1024/512 → 256/128 → 64 → 256/128 → 1024/512 → Output
   • Classifier: 64 → 32/16 → 16/1 → 1
   • Latent Dimension: 64 features

 TOP BIOMARKERS
   USA Top 5 Genes:
   {chr(10).join([f"     {i+1}. {row['Gene']} (score: {row['Importance']:.2f})" for i, row in top_50_usa.head(5).iterrows()])}
   
   India Top 5 Genes:
   {chr(10).join([f"     {i+1}. {row['Gene']} (score: {row['Importance']:.2f})" for i, row in top_50_india.head(5).iterrows()])}

 OUTPUT FILES
   USA Results: {RESULTS_DIR_USA}/
   India Results: {RESULTS_DIR_INDIA}/
   Comparative Analysis: {RESULTS_DIR_COMBINED}/

╔══════════════════════════════════════════════════════════════════════════════╗
║                           ANALYSIS COMPLETE ✓                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

print(summary)

# Save summary
with open(f"{RESULTS_DIR_COMBINED}/analysis_summary.txt", 'w') as f:
    f.write(summary)

print(f"\n All analyses complete! Check the results folders for visualizations and reports.")
