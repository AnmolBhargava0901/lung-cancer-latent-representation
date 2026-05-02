# ==========================================
# LUNG CANCER GENOMIC AI PIPELINE (FIXED & UPGRADED)
# Autoencoder + Latent Classifier + Explainable AI
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA

# 1. SETUP & CONFIGURATION
# ==========================================
INDIA_DATASET_PATH = 'usa_lung_cancer_ml_ready.csv'
RESULTS_DIR = 'lung_cancer_results'
SEED = 42

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Set randomness for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f" Libraries Loaded. Results will be saved to: {RESULTS_DIR}/")

# 2. DATA LOADING & CLEANING
# ==========================================
def load_and_preprocess(filepath):
    print(f"\n Loading Data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Identify target column (assuming last column or specific names)
    target_col = df.columns[-1]
    if 'Label' in df.columns: target_col = 'Label'
    elif 'Class' in df.columns: target_col = 'Class'
    
    # Drop ID columns and Target
    drop_cols = [target_col]
    if 'Sample_ID' in df.columns: drop_cols.append('Sample_ID')
    
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    # Store gene names for later
    gene_names = X.columns.tolist()
    
    # Encode Labels (Normal=0, Cancer=1)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Scale Data (Standardization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"   -> Features: {X_scaled.shape[1]} genes")
    print(f"   -> Samples: {X_scaled.shape[0]}")
    print(f"   -> Class Balance: {np.unique(y_enc, return_counts=True)}")
    
    return X_scaled, y_enc, gene_names

# Load Data
X, y, gene_names = load_and_preprocess(INDIA_DATASET_PATH)

# Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, stratify=y, random_state=SEED
)

# 3. AUTOENCODER MODEL (Unsupervised)
# ==========================================
INPUT_DIM = X_train.shape[1]
LATENT_DIM = 64

input_layer = keras.Input(shape=(INPUT_DIM,))

# Encoder (Compresses Data)
e = layers.Dense(1024, activation='relu')(input_layer)
e = layers.BatchNormalization()(e)
e = layers.Dense(256, activation='relu')(e)

# FIXED: Changed activation to 'tanh' to prevent dead neurons (empty plots)
latent = layers.Dense(LATENT_DIM, activation='tanh', name="Latent_Space")(e)

# Decoder (Reconstructs Data)
d = layers.Dense(256, activation='relu')(latent)
d = layers.Dense(1024, activation='relu')(d)
output_layer = layers.Dense(INPUT_DIM, activation='linear')(d)

# Compile Models
autoencoder = keras.Model(input_layer, output_layer, name="Autoencoder")
encoder = keras.Model(input_layer, latent, name="Encoder")

autoencoder.compile(optimizer='adam', loss='mse')

# 4. TRAIN AUTOENCODER
# ==========================================
print("\n Training Autoencoder (Pattern Discovery)...")
history_ae = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, X_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

# Save Training History Plot
plt.figure(figsize=(8,4))
plt.plot(history_ae.history['loss'], label='Train Loss')
plt.plot(history_ae.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss (Reconstruction Error)')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(f"{RESULTS_DIR}/1_training_loss.png")
plt.close()

# 5. LATENT SPACE VISUALIZATION (PCA)
# ==========================================
print("\n Generating Latent Space Visualization...")
X_test_latent = encoder.predict(X_test)

# Check for dead neurons
if np.all(X_test_latent == 0):
    print(" WARNING: Latent vector is all zeros! Model collapsed.")
else:
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(X_test_latent)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=latent_2d[:,0], y=latent_2d[:,1], hue=y_test, palette='coolwarm', s=100, alpha=0.8)
    plt.title("Latent Space: Separation of Cancer vs Normal")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Class (0=Norm, 1=Cancer)")
    plt.savefig(f"{RESULTS_DIR}/2_latent_space_pca.png")
    plt.show()

# 6. CLASSIFIER (Supervised Learning on Latent Features)
# ==========================================
print("\n Training Classifier on Latent Features...")

X_train_latent = encoder.predict(X_train)

clf_input = keras.Input(shape=(LATENT_DIM,))
c = layers.Dense(32, activation='relu')(clf_input)
c = layers.Dropout(0.3)(c)
c = layers.Dense(16, activation='relu')(c)
clf_output = layers.Dense(1, activation='sigmoid')(c)

classifier = keras.Model(clf_input, clf_output)
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(
    X_train_latent, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_latent, y_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)

# 7. EVALUATION & METRICS
# ==========================================
y_prob = classifier.predict(X_test_latent).ravel()
y_pred = (y_prob > 0.5).astype(int)

# Classification Report
print("\n CLASSIFICATION RESULTS:")
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc:.4f}")

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Cancer'], yticklabels=['Normal', 'Cancer'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{RESULTS_DIR}/3_confusion_matrix.png")
plt.close()

# 8. GENE IMPORTANCE (Explainability)
# ==========================================
print("\n Extracting Top Biomarkers...")

# We look at the weights connecting Input -> First Hidden Layer
w_enc = autoencoder.layers[1].get_weights()[0] # First Dense Layer
importance_scores = np.sum(np.abs(w_enc), axis=1)

# Create DataFrame
importance_df = pd.DataFrame({
    "Gene": gene_names,
    "Importance": importance_scores
}).sort_values(by="Importance", ascending=False)

# Save Top 50 Genes to CSV
top_50 = importance_df.head(50)
top_50.to_csv(f"{RESULTS_DIR}/top_50_genes.csv", index=False)
print(f"   -> Top 50 Genes saved to {RESULTS_DIR}/top_50_genes.csv")

# Plot Top 20 Genes
plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Gene", data=top_50.head(20), palette="viridis")
plt.title("Top 20 Genes Driving the Cancer Diagnosis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Gene ID")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/4_gene_importance.png")
plt.show()

print("\n PROJECT COMPLETE. All results saved in 'lung_cancer_results/' folder.")
