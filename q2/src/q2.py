import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# A. Load Data, Preprocessing, Scaling, and PCA
# ---------------------------------------------------------
print("--- Part A: Preprocessing ---")

# 1. Load Dataset
df = pd.read_csv(r'q2\\data\\Mall_Customers.csv')
print(f"Dataset Loaded. Shape: {df.shape}")

# 2. Cleaning (Checking for nulls)
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print("NaN values dropped.")
else:
    print("No NaN values found.")

# 3. Select Numerical Features
# Based on the CSV structure: Age, Annual Income (k$), Spending Score (1-100)
# CustomerID is an ID, Gender is categorical.
feature_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X_original = df[feature_cols]

# 4. Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)
print("Features scaled using StandardScaler.")

# 5. PCA for Visualization (2D)
pca = PCA(n_components=2, random_state=23)
X_pca = pca.fit_transform(X_scaled)
print("PCA (2D) created for visualization.")
print("-" * 30)

# ---------------------------------------------------------
# B. K-Means Clustering
# ---------------------------------------------------------
print("\n--- Part B: K-Means Clustering ---")
inertia_list = []
silhouette_list = []
k_range = range(2, 11)

print(f"{'K':<5} | {'Inertia':<15} | {'Silhouette Score'}")
print("-" * 40)

best_kmeans_k = 2
best_kmeans_score = -1

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=23, n_init=10)
    kmeans.fit(X_scaled)
    
    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    
    inertia_list.append(inertia)
    silhouette_list.append(sil_score)
    
    print(f"{k:<5} | {inertia:<15.4f} | {sil_score:.4f}")
    
    # Logic to select best K based on max silhouette (for automation in next steps)
    if sil_score > best_kmeans_score:
        best_kmeans_score = sil_score
        best_kmeans_k = k

print(f"\nSelected Best K (based on max Silhouette): {best_kmeans_k}")
print("-" * 30)

# ---------------------------------------------------------
# C. Agglomerative Clustering
# ---------------------------------------------------------
print("\n--- Part C: Agglomerative Clustering ---")
linkages = ['single', 'complete', 'average', 'ward']
best_linkage = ''
best_agg_score = -1

print(f"Using K = {best_kmeans_k} (selected from Part B)")
print(f"{'Linkage':<15} | {'Silhouette Score'}")
print("-" * 40)

for linkage in linkages:
    agg = AgglomerativeClustering(n_clusters=best_kmeans_k, linkage=linkage)
    labels = agg.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, labels)
    print(f"{linkage:<15} | {sil_score:.4f}")
    
    if sil_score > best_agg_score:
        best_agg_score = sil_score
        best_linkage = linkage

print(f"\nSelected Best Linkage: {best_linkage}")
print("-" * 30)

# ---------------------------------------------------------
# D. DBSCAN Clustering
# ---------------------------------------------------------
print("\n--- Part D: DBSCAN Clustering ---")
eps_values = [0.2, 0.4, 0.6, 0.8, 1.0]
min_samples_values = [3, 5, 10]

best_dbscan_score = -1
best_dbscan_params = (0, 0)

print(f"{'Eps':<5} | {'MinPts':<6} | {'Clusters':<8} | {'Noise %':<8} | {'Sil (No Noise)'}")
print("-" * 60)

for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        # Silhouette on non-noise points
        if n_clusters > 1:
            mask = labels != -1
            if sum(mask) > 1: # Need at least 2 points to calc silhouette
                sil_score = silhouette_score(X_scaled[mask], labels[mask])
            else:
                sil_score = -1.0
        else:
            sil_score = -1.0 # Invalid for single cluster or only noise
            
        print(f"{eps:<5} | {min_samples:<6} | {n_clusters:<8} | {noise_ratio:.2%}   | {sil_score:.4f}")
        
        # Update best (Criteria: Valid clusters (>1) and max silhouette)
        if n_clusters > 1 and sil_score > best_dbscan_score:
            best_dbscan_score = sil_score
            best_dbscan_params = (eps, min_samples)

print(f"\nSelected Best DBSCAN Params: Eps={best_dbscan_params[0]}, MinPts={best_dbscan_params[1]}")
print("-" * 30)

# ---------------------------------------------------------
# E. Visualization
# ---------------------------------------------------------
print("\n--- Part E: Visualization ---")

# 1. Fit Best Models
# K-Means
kmeans_final = KMeans(n_clusters=best_kmeans_k, random_state=23, n_init=10)
labels_kmeans = kmeans_final.fit_predict(X_scaled)

# Agglomerative
agg_final = AgglomerativeClustering(n_clusters=best_kmeans_k, linkage=best_linkage)
labels_agg = agg_final.fit_predict(X_scaled)

# DBSCAN
db_final = DBSCAN(eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
labels_db = db_final.fit_predict(X_scaled)

# 2. Plotting
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Helper function for plotting
def plot_clusters(ax, labels, title):
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'k'  # Black for noise
            label_text = "Noise"
            marker = 'x'
        else:
            label_text = f"Cluster {k}"
            marker = 'o'
            
        class_member_mask = (labels == k)
        xy = X_pca[class_member_mask]
        ax.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=col, markeredgecolor='k', markersize=8, label=label_text)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    # ax.legend() # Legend might be too crowded, can uncomment if needed

# Plot 1: K-Means
plot_clusters(axes[0], labels_kmeans, f'K-Means (K={best_kmeans_k})')

# Plot 2: Agglomerative
plot_clusters(axes[1], labels_agg, f'Agglomerative ({best_linkage})')

# Plot 3: DBSCAN
plot_clusters(axes[2], labels_db, f'DBSCAN (eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]})')

plt.suptitle('Comparison of Clustering Algorithms on Mall Customers (PCA Space)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Visualization plot generated.")
