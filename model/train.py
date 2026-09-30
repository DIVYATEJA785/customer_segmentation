import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Feature column definitions
NUMERICAL_FEATURES = [
    'Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency',
    'Average_Order_Value', 'Total_Purchases', 'Recency',
    'Online_Purchases', 'Offline_Purchases', 'Website_Visits'
]
CATEGORICAL_FEATURES = ['Gender']
FEATURE_COLUMNS = ['Gender'] + NUMERICAL_FEATURES

def encode_gender(series):
    """Encodes Gender: Female -> 0, Male -> 1."""
    return series.astype(str).str.strip().str.capitalize().map({'Female': 0, 'Male': 1}).fillna(0).astype(int)

def determine_segment_name(row):
    """
    Dynamically generates meaningful customer segment titles and actionable descriptions
    based on the cluster's average statistical attributes.
    """
    income = row.get('Annual_Income', 0)
    spending = row.get('Spending_Score', 0)
    age = row.get('Age', 0)
    recency = row.get('Recency', 0)
    freq = row.get('Purchase_Frequency', 0)
    aov = row.get('Average_Order_Value', 0)
    online = row.get('Online_Purchases', 0)

    # Dynamic classification rules based on relative centroids (in INR ₹)
    if income >= 1500000 and spending >= 60:
        name = "High-Value Champions"
        badge = "badge-success"
        desc = "Affluent customers with high spending, top order value, and frequent transactions."
        strategy = "Provide VIP loyalty rewards, exclusive early access, dedicated concierge, and premium bundles."
    elif income >= 1500000 and spending < 45:
        name = "Affluent Conservative Savers"
        badge = "badge-info"
        desc = "High-earning customers who spend conservatively with high basket size per purchase."
        strategy = "Offer high-ticket value packages, premium quality warranties, and milestone savings incentives."
    elif income < 1200000 and spending >= 65 and age < 35:
        name = "Young Impulse Trendsetters"
        badge = "badge-primary"
        desc = "Younger tech-savvy shoppers with moderate income and enthusiastic online shopping habits."
        strategy = "Engage through targeted social media campaigns, flash sales, gamified mobile apps, and trending drops."
    elif recency >= 100:
        name = "At-Risk Occasional Customers"
        badge = "badge-warning"
        desc = "Moderate-income shoppers showing declining engagement, infrequent orders, and high dormancy."
        strategy = "Deploy automated win-back emails, personalized reactivation coupons, and customer feedback surveys."
    elif income < 700000 and spending < 50:
        name = "Budget-Conscious Shoppers"
        badge = "badge-secondary"
        desc = "Price-sensitive buyers seeking seasonal discounts, clearance promotions, and essential goods."
        strategy = "Target with volume discounts, free-shipping thresholds, and affordable entry-level products."
    else:
        name = "Steady Core Customers"
        badge = "badge-dark"
        desc = "Reliable mainstream customers exhibiting steady, average purchasing patterns across channels."
        strategy = "Maintain engagement with seasonal newsletters, points-based rewards, and cross-category discovery."

    return name, desc, strategy, badge

def train_and_evaluate(dataset_path='dataset/customers.csv', output_dir='outputs', model_dir='model', selected_k=5):
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"--> [Step 1/11] Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"    Loaded {len(df)} records with {len(df.columns)} columns.")

    print("--> [Step 2/11] Preprocessing & Cleaning")
    # Check missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"    Found {missing_count} missing values. Imputing with median/mode...")
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
    else:
        print("    No missing values detected.")

    # Remove duplicates
    initial_len = len(df)
    subset_cols = [c for c in df.columns if c != 'Customer_ID']
    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    if len(df) < initial_len:
        print(f"    Removed {initial_len - len(df)} duplicate records.")

    # Encode Categorical features
    df['Gender_Encoded'] = encode_gender(df['Gender'])

    # Feature matrix
    feature_cols = ['Gender_Encoded'] + NUMERICAL_FEATURES
    X = df[feature_cols].copy()

    print("--> [Step 3/11] Feature Scaling with StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"    Scaler saved to: {scaler_path}")

    print("--> [Step 4/11] Testing K values from 2 to 10 for Optimal K (Elbow & Silhouette)")
    k_range = range(2, 11)
    wcss = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        km.fit(X_scaled)
        labels = km.labels_
        wcss.append(float(km.inertia_))
        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))
        ch = float(calinski_harabasz_score(X_scaled, labels))
        silhouette_scores.append(sil)
        davies_bouldin_scores.append(db)
        calinski_scores.append(ch)
        print(f"    K={k:2d} | WCSS: {km.inertia_:10.2f} | Silhouette: {sil:.4f} | DB Index: {db:.4f} | CH Score: {ch:8.2f}")

    final_k = selected_k if selected_k is not None else 5
    print(f"--> [Step 5/11] Selected Cluster Count: K = {final_k}")

    # 1. Elbow Curve
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), wcss, marker='o', color='#4f46e5', linewidth=2.5, markersize=8)
    plt.axvline(x=final_k, color='#ef4444', linestyle='--', label=f'Selected K={final_k}')
    plt.title('Elbow Method For Optimal K (WCSS vs Clusters)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'elbow_curve.png'), dpi=200)
    plt.close()

    # 2. Silhouette Graph
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), silhouette_scores, marker='s', color='#06b6d4', linewidth=2.5, markersize=8)
    plt.axvline(x=final_k, color='#ef4444', linestyle='--', label=f'Selected K={final_k}')
    plt.title('Silhouette Score vs Number of Clusters', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Silhouette Score', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'silhouette_graph.png'), dpi=200)
    plt.close()

    print(f"--> [Step 6/11] Training Final K-Means Model with K={final_k}")
    kmeans = KMeans(n_clusters=final_k, init='k-means++', n_init=25, max_iter=500, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # Save model
    model_path = os.path.join(model_dir, 'kmeans_model.pkl')
    joblib.dump(kmeans, model_path)
    print(f"    Trained model saved to: {model_path}")

    print("--> [Step 7/11] Cluster Profiling & Dynamic Segment Interpretation")
    summary_cols = NUMERICAL_FEATURES
    cluster_means = df.groupby('Cluster_ID')[summary_cols].mean().round(2)
    cluster_counts = df['Cluster_ID'].value_counts().sort_index()

    segment_info = {}
    for cid in range(final_k):
        mean_row = cluster_means.loc[cid].to_dict()
        s_name, s_desc, s_strat, s_badge = determine_segment_name(mean_row)
        segment_info[int(cid)] = {
            'Cluster_ID': int(cid),
            'Segment_Name': s_name,
            'Description': s_desc,
            'Marketing_Strategy': s_strat,
            'Badge': s_badge,
            'Customer_Count': int(cluster_counts[cid]),
            'Percentage': round((cluster_counts[cid] / len(df)) * 100, 2),
            **mean_row
        }

    # Map segment names into dataframe
    df['Segment_Name'] = df['Cluster_ID'].map(lambda cid: segment_info[cid]['Segment_Name'])

    # Save summary dataframe
    summary_df = pd.DataFrame(list(segment_info.values()))
    summary_csv_path = os.path.join(output_dir, 'cluster_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"    Cluster summary saved to: {summary_csv_path}")

    # Save clustered customers
    clustered_csv_path = os.path.join(output_dir, 'clustered_customers.csv')
    clean_export_df = df.drop(columns=['Gender_Encoded'])
    clean_export_df.to_csv(clustered_csv_path, index=False)
    print(f"    Clustered customer dataset saved to: {clustered_csv_path}")

    print("--> [Step 8/11] Generating Comprehensive Visualizations")
    palette = sns.color_palette("tab10", final_k)

    # 3. Cluster Distribution Bar Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        [f"Cluster {cid}\n{segment_info[cid]['Segment_Name']}" for cid in range(final_k)],
        cluster_counts.values,
        color=palette,
        edgecolor='#1e293b',
        linewidth=1.2
    )
    plt.title('Customer Count Across Identified Clusters', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Number of Customers', fontsize=11)
    plt.xticks(fontsize=9, rotation=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 4, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cluster_distribution.png'), dpi=200)
    plt.close()

    # 4. Income vs Spending Score
    plt.figure(figsize=(9, 6))
    for cid in range(final_k):
        sub = df[df['Cluster_ID'] == cid]
        plt.scatter(
            sub['Annual_Income'], sub['Spending_Score'],
            label=f"Cluster {cid}: {segment_info[cid]['Segment_Name']}",
            color=palette[cid], alpha=0.75, edgecolors='black', linewidth=0.5, s=60
        )
    plt.title('Customer Segments: Annual Income vs Spending Score', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Annual Income (₹)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'income_vs_spending.png'), dpi=200)
    plt.close()

    # 5. Age vs Spending Score
    plt.figure(figsize=(9, 6))
    for cid in range(final_k):
        sub = df[df['Cluster_ID'] == cid]
        plt.scatter(
            sub['Age'], sub['Spending_Score'],
            label=f"Cluster {cid}: {segment_info[cid]['Segment_Name']}",
            color=palette[cid], alpha=0.75, edgecolors='black', linewidth=0.5, s=60
        )
    plt.title('Customer Segments: Age vs Spending Score', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Age (Years)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'age_vs_spending.png'), dpi=200)
    plt.close()

    # 6. Purchase Frequency vs Spending Score
    plt.figure(figsize=(9, 6))
    for cid in range(final_k):
        sub = df[df['Cluster_ID'] == cid]
        plt.scatter(
            sub['Purchase_Frequency'], sub['Spending_Score'],
            label=f"Cluster {cid}: {segment_info[cid]['Segment_Name']}",
            color=palette[cid], alpha=0.75, edgecolors='black', linewidth=0.5, s=60
        )
    plt.title('Customer Segments: Purchase Frequency vs Spending Score', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Purchase Frequency (Orders/Month)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'frequency_vs_spending.png'), dpi=200)
    plt.close()

    # 7. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[NUMERICAL_FEATURES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=200)
    plt.close()

    # 8. 2D PCA Cluster Visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    plt.figure(figsize=(9, 6))
    for cid in range(final_k):
        sub = df[df['Cluster_ID'] == cid]
        plt.scatter(
            sub['PCA1'], sub['PCA2'],
            label=f"Cluster {cid}: {segment_info[cid]['Segment_Name']}",
            color=palette[cid], alpha=0.75, edgecolors='black', linewidth=0.5, s=60
        )
    plt.title(f'2D PCA Projection of Customer Clusters (Variance Explained: {sum(pca.explained_variance_ratio_)*100:.1f}%)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Principal Component 1', fontsize=11)
    plt.ylabel('Principal Component 2', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pca_2d_clusters.png'), dpi=200)
    plt.close()

    # 9. Boxplot of Key Metrics per Cluster (warning-free)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    key_metrics = ['Annual_Income', 'Spending_Score', 'Purchase_Frequency', 'Recency']
    for idx, col in enumerate(key_metrics):
        r, c = divmod(idx, 2)
        sns.boxplot(ax=axes[r, c], x='Cluster_ID', y=col, data=df, hue='Cluster_ID', palette=palette, legend=False)
        axes[r, c].set_title(f'{col.replace("_", " ")} Distribution by Cluster', fontweight='bold')
        axes[r, c].set_xlabel('Cluster ID')
        axes[r, c].grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cluster_boxplots.png'), dpi=200)
    plt.close()

    print("--> [Step 9/11] Computing Final Clustering Evaluation Metrics")
    final_sil = float(silhouette_score(X_scaled, df['Cluster_ID']))
    final_db = float(davies_bouldin_score(X_scaled, df['Cluster_ID']))
    final_ch = float(calinski_harabasz_score(X_scaled, df['Cluster_ID']))
    final_wcss = float(kmeans.inertia_)

    metrics = {
        'total_customers': int(len(df)),
        'num_clusters': int(final_k),
        'silhouette_score': round(final_sil, 4),
        'davies_bouldin_index': round(final_db, 4),
        'calinski_harabasz_score': round(final_ch, 2),
        'wcss': round(final_wcss, 2),
        'elbow_k_values': list(k_range),
        'elbow_wcss': [round(x, 2) for x in wcss],
        'silhouette_per_k': [round(x, 4) for x in silhouette_scores],
        'davies_bouldin_per_k': [round(x, 4) for x in davies_bouldin_scores],
        'calinski_per_k': [round(x, 2) for x in calinski_scores],
        'avg_income': round(float(df['Annual_Income'].mean()), 2),
        'avg_spending': round(float(df['Spending_Score'].mean()), 2),
        'avg_age': round(float(df['Age'].mean()), 2),
        'avg_order_value': round(float(df['Average_Order_Value'].mean()), 2),
        'avg_recency': round(float(df['Recency'].mean()), 2),
        'gender_distribution': df['Gender'].value_counts().to_dict()
    }

    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"    Metrics saved to: {metrics_path}")

    print("--> [Step 10/11] Saving Model Metadata")
    metadata = {
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'feature_columns': feature_cols,
        'k': final_k,
        'segments': segment_info,
        'metrics': metrics
    }
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"    Metadata saved to: {metadata_path}")

    print("--> [Step 11/11] Training Pipeline Completed Successfully!")
    print(f"    - Clusters: {final_k}")
    print(f"    - Silhouette Score: {final_sil:.4f}")
    print(f"    - Davies-Bouldin Index: {final_db:.4f}")
    print(f"    - Calinski-Harabasz Score: {final_ch:.2f}")

    return {
        'df': df,
        'kmeans': kmeans,
        'scaler': scaler,
        'metrics': metrics,
        'segments': segment_info
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Customer Segmentation Training Pipeline")
    parser.add_argument('--dataset', type=str, default='dataset/customers.csv', help='Path to dataset CSV')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters (default: 5)')
    args = parser.parse_args()

    train_and_evaluate(dataset_path=args.dataset, selected_k=args.k)
