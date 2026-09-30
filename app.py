import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from model.train import (
    train_and_evaluate,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    encode_gender,
    determine_segment_name
)

app = Flask(__name__)
app.secret_key = "customer_segmentation_btech_secret_key"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'customers.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
KMEANS_PATH = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'model_metadata.pkl')
CLUSTERED_DATA_PATH = os.path.join(OUTPUTS_DIR, 'clustered_customers.csv')
SUMMARY_DATA_PATH = os.path.join(OUTPUTS_DIR, 'cluster_summary.csv')
METRICS_PATH = os.path.join(OUTPUTS_DIR, 'metrics.json')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_system_state():
    """Safely loads models, metadata, and dataset. Trains if missing."""
    if not os.path.exists(DATASET_PATH):
        from generate_dataset import generate_customer_dataset
        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        df_gen = generate_customer_dataset(1000)
        df_gen.to_csv(DATASET_PATH, index=False)

    if not (os.path.exists(KMEANS_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METADATA_PATH)):
        train_and_evaluate(dataset_path=DATASET_PATH, output_dir=OUTPUTS_DIR, model_dir=MODEL_DIR, selected_k=5)

    kmeans = joblib.load(KMEANS_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = joblib.load(METADATA_PATH)
    
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    summary_df = pd.read_csv(SUMMARY_DATA_PATH)
    clustered_df = pd.read_csv(CLUSTERED_DATA_PATH)

    return kmeans, scaler, metadata, metrics, summary_df, clustered_df

# Initialize state
try:
    kmeans_model, scaler_model, model_metadata, system_metrics, summary_data, clustered_data = load_system_state()
except Exception as e:
    print(f"Initialization Warning: {e}")
    kmeans_model, scaler_model, model_metadata, system_metrics, summary_data, clustered_data = None, None, None, {}, pd.DataFrame(), pd.DataFrame()

# Serve generated plots directly
@app.route('/outputs/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

@app.route('/')
def home():
    global system_metrics, summary_data
    if system_metrics is None or len(system_metrics) == 0:
        try:
            _, _, _, system_metrics, summary_data, _ = load_system_state()
        except Exception:
            pass
    return render_template('index.html', metrics=system_metrics, segments=summary_data.to_dict('records') if not summary_data.empty else [])

@app.route('/dashboard')
def dashboard():
    global system_metrics, summary_data, clustered_data
    try:
        if system_metrics is None or len(system_metrics) == 0:
            _, _, _, system_metrics, summary_data, clustered_data = load_system_state()
    except Exception as e:
        flash(f"Error loading dashboard data: {str(e)}", "danger")

    return render_template(
        'dashboard.html',
        metrics=system_metrics,
        summary=summary_data.to_dict('records') if not summary_data.empty else [],
        total_customers=len(clustered_data) if not clustered_data.empty else 0
    )

@app.route('/dataset')
def dataset_view():
    try:
        if not os.path.exists(DATASET_PATH):
            flash("Dataset file not found. Generating default dataset...", "warning")
            from generate_dataset import generate_customer_dataset
            df_gen = generate_customer_dataset(1000)
            df_gen.to_csv(DATASET_PATH, index=False)

        df = pd.read_csv(DATASET_PATH)
        stats = df.describe().round(2).to_dict()
        columns = list(df.columns)
        null_counts = df.isnull().sum().to_dict()
        duplicates = int(df.duplicated().sum())
        total_rows = len(df)
        total_cols = len(columns)
        preview_data = df.head(50).to_dict('records')
        gender_counts = df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {}

        return render_template(
            'dataset.html',
            columns=columns,
            total_rows=total_rows,
            total_cols=total_cols,
            null_counts=null_counts,
            duplicates=duplicates,
            stats=stats,
            preview=preview_data,
            gender_counts=gender_counts
        )
    except Exception as e:
        flash(f"Error reading dataset: {str(e)}", "danger")
        return render_template('dataset.html', columns=[], total_rows=0, total_cols=0, null_counts={}, duplicates=0, stats={}, preview=[], gender_counts={})

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global kmeans_model, scaler_model, model_metadata
    prediction_result = None

    if request.method == 'POST':
        try:
            if kmeans_model is None or scaler_model is None or model_metadata is None:
                kmeans_model, scaler_model, model_metadata, _, _, _ = load_system_state()

            # Extract and validate form inputs
            gender_val = request.form.get('gender', 'Female').strip()
            age_val = float(request.form.get('age', 0))
            income_val = float(request.form.get('income', 0))
            spending_val = float(request.form.get('spending', 0))
            freq_val = float(request.form.get('frequency', 0))
            aov_val = float(request.form.get('aov', 0))
            total_purchases_val = float(request.form.get('total_purchases', 0))
            recency_val = float(request.form.get('recency', 0))
            online_val = float(request.form.get('online', 0))
            offline_val = float(request.form.get('offline', 0))
            visits_val = float(request.form.get('visits', 0))

            # Validate range constraints
            if not (15 <= age_val <= 100):
                flash("Please enter a realistic age between 15 and 100.", "warning")
                return render_template('prediction.html', prediction=None)
            if income_val < 50000:
                flash("Annual income should be at least ₹50,000.", "warning")
                return render_template('prediction.html', prediction=None)
            if not (1 <= spending_val <= 100):
                flash("Spending score must be between 1 and 100.", "warning")
                return render_template('prediction.html', prediction=None)

            # Encode Gender
            gender_encoded = 1 if gender_val.lower() == 'male' else 0

            # Feature DataFrame matching exact feature names used during training:
            feature_cols = ['Gender_Encoded', 'Age', 'Annual_Income', 'Spending_Score', 'Purchase_Frequency',
                            'Average_Order_Value', 'Total_Purchases', 'Recency', 'Online_Purchases',
                            'Offline_Purchases', 'Website_Visits']
            raw_features = pd.DataFrame([[
                gender_encoded, age_val, income_val, spending_val, freq_val,
                aov_val, total_purchases_val, recency_val, online_val, offline_val, visits_val
            ]], columns=feature_cols)

            # Scale using fitted StandardScaler
            scaled_features = scaler_model.transform(raw_features)

            # Predict cluster
            cluster_id = int(kmeans_model.predict(scaled_features)[0])

            # Retrieve cluster details
            segments_info = model_metadata.get('segments', {})
            cluster_info = segments_info.get(cluster_id, {})

            # Distance to cluster centroid
            centroid = kmeans_model.cluster_centers_[cluster_id]
            dist_to_center = float(np.linalg.norm(scaled_features - centroid))

            prediction_result = {
                'cluster_id': cluster_id,
                'segment_name': cluster_info.get('Segment_Name', f'Cluster {cluster_id}'),
                'description': cluster_info.get('Description', 'Identified Customer Segment'),
                'strategy': cluster_info.get('Marketing_Strategy', 'Apply personalized engagement strategy.'),
                'badge': cluster_info.get('Badge', 'badge-primary'),
                'customer_count': cluster_info.get('Customer_Count', 'N/A'),
                'percentage': cluster_info.get('Percentage', 'N/A'),
                'cluster_stats': {
                    'avg_income': cluster_info.get('Annual_Income', income_val),
                    'avg_spending': cluster_info.get('Spending_Score', spending_val),
                    'avg_age': cluster_info.get('Age', age_val),
                    'avg_aov': cluster_info.get('Average_Order_Value', aov_val),
                    'avg_recency': cluster_info.get('Recency', recency_val)
                },
                'input_data': {
                    'Gender': gender_val,
                    'Age': int(age_val),
                    'Annual_Income': f"₹{income_val:,.2f}",
                    'Spending_Score': int(spending_val),
                    'Purchase_Frequency': int(freq_val),
                    'Average_Order_Value': f"₹{aov_val:,.2f}",
                    'Total_Purchases': int(total_purchases_val),
                    'Recency': f"{int(recency_val)} days",
                    'Online_Purchases': int(online_val),
                    'Offline_Purchases': int(offline_val),
                    'Website_Visits': int(visits_val)
                },
                'distance': round(dist_to_center, 3)
            }
            flash(f"Successfully classified customer into Cluster {cluster_id}: {prediction_result['segment_name']}!", "success")

        except ValueError as ve:
            flash(f"Invalid numerical input: {str(ve)}. Please verify all input fields.", "danger")
        except Exception as e:
            flash(f"Prediction Error: {str(e)}", "danger")

    return render_template('prediction.html', prediction=prediction_result)

@app.route('/results')
def results():
    global clustered_data, summary_data, system_metrics
    try:
        if clustered_data is None or clustered_data.empty:
            _, _, _, system_metrics, summary_data, clustered_data = load_system_state()

        # Handle filter by cluster
        cluster_filter = request.args.get('cluster', 'all')
        if cluster_filter != 'all':
            filtered_df = clustered_data[clustered_data['Cluster_ID'] == int(cluster_filter)]
        else:
            filtered_df = clustered_data

        # Limit to first 100 rows for smooth browser rendering
        records = filtered_df.head(100).to_dict('records')
        total_records = len(filtered_df)
        all_clusters = sorted(clustered_data['Cluster_ID'].unique().tolist()) if not clustered_data.empty else []

        return render_template(
            'results.html',
            records=records,
            total_records=total_records,
            all_clusters=all_clusters,
            selected_cluster=cluster_filter,
            summary=summary_data.to_dict('records') if not summary_data.empty else [],
            metrics=system_metrics
        )
    except Exception as e:
        flash(f"Error loading results: {str(e)}", "danger")
        return render_template('results.html', records=[], total_records=0, all_clusters=[], selected_cluster='all', summary=[], metrics={})

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global kmeans_model, scaler_model, model_metadata, system_metrics, summary_data, clustered_data

    if 'file' not in request.files:
        flash("No file part in the upload request.", "danger")
        return redirect(url_for('dataset_view'))

    file = request.files['file']
    if file.filename == '':
        flash("No file selected for upload.", "warning")
        return redirect(url_for('dataset_view'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(BASE_DIR, 'dataset', filename)

        try:
            file.save(upload_path)
            # Validate CSV
            df_test = pd.read_csv(upload_path)
            if df_test.empty:
                flash("Uploaded CSV file is completely empty.", "danger")
                if os.path.exists(upload_path) and upload_path != DATASET_PATH:
                    os.remove(upload_path)
                return redirect(url_for('dataset_view'))

            # Check required columns
            required_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
            missing_cols = [c for c in required_cols if c not in df_test.columns]
            if missing_cols:
                flash(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}", "danger")
                if os.path.exists(upload_path) and upload_path != DATASET_PATH:
                    os.remove(upload_path)
                return redirect(url_for('dataset_view'))

            # Overwrite default dataset and retrain
            df_test.to_csv(DATASET_PATH, index=False)
            flash("Dataset uploaded and verified successfully! Retraining models...", "info")

            # Retrain
            selected_k = int(request.form.get('k_clusters', 5))
            train_and_evaluate(dataset_path=DATASET_PATH, output_dir=OUTPUTS_DIR, model_dir=MODEL_DIR, selected_k=selected_k)

            # Reload memory state
            kmeans_model, scaler_model, model_metadata, system_metrics, summary_data, clustered_data = load_system_state()
            flash("Clustering pipeline executed successfully on uploaded dataset!", "success")
            return redirect(url_for('dashboard'))

        except Exception as e:
            flash(f"Error processing uploaded CSV: {str(e)}", "danger")
            return redirect(url_for('dataset_view'))
    else:
        flash("Invalid file type. Only .csv files are supported.", "danger")
        return redirect(url_for('dataset_view'))

@app.route('/download/<filename>')
def download_file(filename):
    allowed_downloads = {
        'clustered_customers.csv': CLUSTERED_DATA_PATH,
        'cluster_summary.csv': SUMMARY_DATA_PATH,
        'customers.csv': DATASET_PATH
    }

    if filename in allowed_downloads and os.path.exists(allowed_downloads[filename]):
        return send_file(
            allowed_downloads[filename],
            as_attachment=True,
            download_name=filename
        )
    else:
        flash("Requested file does not exist or cannot be downloaded.", "danger")
        return redirect(url_for('results'))

@app.route('/api/cluster_data')
def api_cluster_data():
    """Provides JSON data for interactive frontend visualizations."""
    global clustered_data, summary_data, system_metrics
    try:
        if clustered_data is None or clustered_data.empty:
            _, _, _, system_metrics, summary_data, clustered_data = load_system_state()

        chart_data = {
            'cluster_counts': summary_data[['Cluster_ID', 'Segment_Name', 'Customer_Count', 'Percentage']].to_dict('records'),
            'elbow_k': system_metrics.get('elbow_k_values', []),
            'elbow_wcss': system_metrics.get('elbow_wcss', []),
            'silhouette_k': system_metrics.get('silhouette_per_k', []),
            'scatter_sample': clustered_data[['Customer_ID', 'Annual_Income', 'Spending_Score', 'Cluster_ID', 'Segment_Name', 'Age']].sample(min(300, len(clustered_data))).to_dict('records')
        }
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("==================================================")
    print(" Customer Segmentation Web Application Starting")
    print(" Open URL: http://127.0.0.1:5000 in your browser")
    print("==================================================")
    app.run(debug=True, host='127.0.0.1', port=5000)
