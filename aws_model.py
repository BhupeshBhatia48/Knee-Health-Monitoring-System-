# aws_knee_health_pipeline.py
# -------------------------------------------------------
# AWS Cloud-Ready End-to-End Pipeline for Knee Health Classification
# Single execution script that:
# 1. Fetches data from AWS S3
# 2. Processes and combines datasets
# 3. Trains model
# 4. Saves model back to S3
# -------------------------------------------------------

import os
import re
import io
import numpy as np
import pandas as pd
from typing import List, Tuple
import time
from datetime import datetime
import tempfile

# AWS imports
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, confusion_matrix
)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cloud
import matplotlib.pyplot as plt
import itertools

import tensorflow as tf
from tensorflow.keras import layers, callbacks, Input, Model


# =====================================================
# AWS CONFIGURATION
# =====================================================
class AWSConfig:
    """AWS S3 Configuration"""
    # AWS Credentials (set these as environment variables for security)
    AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID', 'YOUR_ACCESS_KEY')
    AWS_SECRET_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY', 'YOUR_SECRET_KEY')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    
    # S3 Bucket Configuration
    S3_BUCKET_NAME = 'knee-health-ml-bucket'  # Change to your bucket name
    S3_INPUT_PREFIX = 'input_data/'           # Folder for input CSVs
    S3_OUTPUT_PREFIX = 'output_data/'         # Folder for processed data
    S3_MODEL_PREFIX = 'models/'               # Folder for trained models
    
    # Input file names in S3
    CLIMBING_FILE = 'Climbing_Data_2.csv'
    WALKING_FILE = 'Walking_Data_2.csv'


class ModelConfig:
    """Model Training Configuration"""
    SEQ_LEN = 120
    BATCH_SIZE = 16
    EPOCHS = 50
    
    SENSOR_TYPES = [
        'Acceleration_upper_x', 'Acceleration_upper_y', 'Acceleration_upper_z',
        'Velocity_upper_x', 'Velocity_upper_y', 'Velocity_upper_z',
        'Velocity_lower_x', 'Velocity_lower_y', 'Velocity_lower_z',
        'Acceleration_lower_x', 'Acceleration_lower_y', 'Acceleration_lower_z'
    ]


# =====================================================
# AWS S3 HANDLER
# =====================================================
class S3Handler:
    """Handle all S3 operations"""
    
    def __init__(self):
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=AWSConfig.AWS_ACCESS_KEY,
                aws_secret_access_key=AWSConfig.AWS_SECRET_KEY,
                region_name=AWSConfig.AWS_REGION
            )
            print("AWS S3 client initialized successfully")
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            raise
    
    def download_csv_from_s3(self, file_name, prefix=''):
        """Download CSV file from S3 and return as DataFrame"""
        try:
            s3_key = f"{prefix}{file_name}"
            print(f"Downloading from S3: s3://{AWSConfig.S3_BUCKET_NAME}/{s3_key}")
            
            response = self.s3_client.get_object(
                Bucket=AWSConfig.S3_BUCKET_NAME,
                Key=s3_key
            )
            
            df = pd.read_csv(io.BytesIO(response['Body'].read()))
            print(f"Successfully downloaded {file_name}: {df.shape}")
            return df
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                print(f"File not found in S3: {s3_key}")
            else:
                print(f"AWS Error: {e}")
            raise
        except NoCredentialsError:
            print("AWS credentials not found!")
            raise
    
    def upload_csv_to_s3(self, df, file_name, prefix=''):
        """Upload DataFrame as CSV to S3"""
        try:
            s3_key = f"{prefix}{file_name}"
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=AWSConfig.S3_BUCKET_NAME,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            print(f"Uploaded to S3: s3://{AWSConfig.S3_BUCKET_NAME}/{s3_key}")
            
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            raise
    
    def upload_model_to_s3(self, local_model_path, s3_file_name):
        """Upload trained model to S3"""
        try:
            s3_key = f"{AWSConfig.S3_MODEL_PREFIX}{s3_file_name}"
            
            self.s3_client.upload_file(
                local_model_path,
                AWSConfig.S3_BUCKET_NAME,
                s3_key
            )
            print(f"Model uploaded to S3: s3://{AWSConfig.S3_BUCKET_NAME}/{s3_key}")
            
        except Exception as e:
            print(f"Error uploading model: {e}")
            raise
    
    def upload_file_to_s3(self, local_path, s3_file_name, prefix=''):
        """Upload any file to S3"""
        try:
            s3_key = f"{prefix}{s3_file_name}"
            self.s3_client.upload_file(local_path, AWSConfig.S3_BUCKET_NAME, s3_key)
            print(f"File uploaded: s3://{AWSConfig.S3_BUCKET_NAME}/{s3_key}")
        except Exception as e:
            print(f"Error uploading file: {e}")


# =====================================================
# DATA PROCESSING FUNCTIONS
# =====================================================
def clean_sensor_value(value):
    """Remove non-numeric characters, keep only numbers and pipes"""
    if pd.isna(value) or value == '':
        return ''
    
    value_str = str(value)
    if '|' in value_str:
        parts = value_str.split('|')
    elif '\\' in value_str:
        parts = value_str.split('\\')
    else:
        parts = [value_str]
    
    cleaned_parts = []
    for part in parts:
        cleaned = re.sub(r'[^0-9.\-]', '', part.strip())
        if cleaned and cleaned not in ['', '.', '-']:
            try:
                float(cleaned)
                cleaned_parts.append(cleaned)
            except ValueError:
                continue
    
    return '|'.join(cleaned_parts) if cleaned_parts else ''


def clean_dataset(df):
    """Clean dataset by removing non-numeric values"""
    print("Cleaning dataset...")
    reading_cols = [col for col in df.columns if col.startswith('Reading_')]
    
    for col in reading_cols:
        df[col] = df[col].apply(clean_sensor_value)
    
    print(f"Cleaned {len(reading_cols)} reading columns")
    return df


def transform_to_sensor_columns(df):
    """Transform from wide format to sensor columns"""
    print("Transforming data structure...")
    
    reading_cols = [col for col in df.columns if col.startswith('Reading_')]
    other_cols = [col for col in df.columns if not col.startswith('Reading_')]
    
    result_df = df[other_cols].copy()
    sensor_data = {sensor: [] for sensor in ModelConfig.SENSOR_TYPES}
    
    for idx, row in df.iterrows():
        row_sensor_data = {sensor: [] for sensor in ModelConfig.SENSOR_TYPES}
        
        for reading_col in reading_cols:
            reading_value = row[reading_col]
            if pd.isna(reading_value) or reading_value == '':
                continue
            
            values = str(reading_value).split('|')
            values = [v.strip() for v in values if v.strip()]
            
            for i, value in enumerate(values):
                if i < len(ModelConfig.SENSOR_TYPES):
                    row_sensor_data[ModelConfig.SENSOR_TYPES[i]].append(value)
        
        for sensor in ModelConfig.SENSOR_TYPES:
            if row_sensor_data[sensor]:
                sensor_data[sensor].append('|'.join(row_sensor_data[sensor]))
            else:
                sensor_data[sensor].append('')
    
    for sensor in ModelConfig.SENSOR_TYPES:
        result_df[sensor] = sensor_data[sensor]
    
    print(f"Transformed shape: {result_df.shape}")
    return result_df


def combine_datasets(climbing_df, walking_df):
    """Combine climbing and walking datasets"""
    print("Combining datasets...")
    
    climbing_df['activity_type'] = 'climbing'
    walking_df['activity_type'] = 'walking'
    
    if list(climbing_df.columns) != list(walking_df.columns):
        walking_df = walking_df[climbing_df.columns]
    
    combined_df = pd.concat([climbing_df, walking_df], ignore_index=True)
    
    print(f"Combined shape: {combined_df.shape}")
    print(f"Activity distribution:\n{combined_df['activity_type'].value_counts()}")
    
    return combined_df


def encode_labels(df):
    """Encode categorical labels"""
    print("Encoding labels...")
    
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    activity_encoder = LabelEncoder()
    df['activity_type_encoded'] = activity_encoder.fit_transform(df['activity_type'])
    
    label_mapping = dict(zip(label_encoder.classes_, 
                            label_encoder.transform(label_encoder.classes_)))
    activity_mapping = dict(zip(activity_encoder.classes_, 
                               activity_encoder.transform(activity_encoder.classes_)))
    
    print(f"Label encoding: {label_mapping}")
    print(f"Activity encoding: {activity_mapping}")
    
    return df, label_encoder, activity_encoder


# =====================================================
# MODEL TRAINING FUNCTIONS
# =====================================================
class TimedConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(TimedConv1D, self).__init__()
        self.conv = layers.Conv1D(filters, kernel_size, **kwargs)
        self.timing_data = []
        
    def call(self, inputs, training=None):
        start = time.perf_counter()
        result = self.conv(inputs, training=training)
        if training:
            self.timing_data.append(time.perf_counter() - start)
        return result


class TimedGRU(layers.Layer):
    def __init__(self, units, **kwargs):
        super(TimedGRU, self).__init__()
        self.gru = layers.GRU(units, **kwargs)
        self.timing_data = []
        
    def call(self, inputs, training=None):
        start = time.perf_counter()
        result = self.gru(inputs, training=training)
        if training:
            self.timing_data.append(time.perf_counter() - start)
        return result


def parse_pipe_series_to_array(s):
    """Convert pipe-separated string to float array"""
    if pd.isna(s) or s == '':
        return np.array([], dtype=np.float32)
    
    parts = [p.strip() for p in str(s).split('|') if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except:
            continue
    return np.array(vals, dtype=np.float32)


def pad_or_truncate(arr, seq_len):
    """Pad or truncate array to seq_len"""
    if len(arr) == 0:
        return np.zeros(seq_len, dtype=np.float32)
    if len(arr) == seq_len:
        return arr
    if len(arr) > seq_len:
        return arr[:seq_len]
    
    pad_val = arr[-1]
    padded = np.pad(arr, (0, seq_len - len(arr)), 
                   mode='constant', constant_values=pad_val)
    return padded.astype(np.float32)


def build_time_series_tensor(df, seq_len):
    """Build 3D tensor from sensor columns"""
    sensor_cols = [col for col in ModelConfig.SENSOR_TYPES if col in df.columns]
    N = len(df)
    C = len(sensor_cols)
    X_ts = np.zeros((N, seq_len, C), dtype=np.float32)
    
    for i, col in enumerate(sensor_cols):
        arrays = df[col].apply(parse_pipe_series_to_array).apply(
            lambda a: pad_or_truncate(a, seq_len))
        X_ts[:, :, i] = np.stack(arrays.values)
    
    return X_ts, sensor_cols


def standardize_channels(X):
    """Channel-wise standardization"""
    N, T, C = X.shape
    X_std = np.empty_like(X)
    
    for c in range(C):
        vec = X[:, :, c].reshape(-1)
        m = np.mean(vec)
        s = np.std(vec) + 1e-8
        X_std[:, :, c] = (X[:, :, c] - m) / s
    
    return X_std


def build_model(seq_len, n_channels, static_dim, n_classes=3):
    """Build CNN+GRU hybrid model"""
    # Time-series branch
    ts_in = Input(shape=(seq_len, n_channels), name="ts_input")
    
    x = TimedConv1D(64, 3, padding='same', activation='relu')(ts_in)
    x = TimedConv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    x = TimedConv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = TimedGRU(128, return_sequences=True)(x)
    x = TimedGRU(64)(x)
    ts_out = layers.Dropout(0.25)(x)
    
    # Static branch
    st_in = Input(shape=(static_dim,), name="static_input")
    s = layers.Dense(32, activation='relu')(st_in)
    s = layers.Dropout(0.2)(s)
    
    # Fusion
    h = layers.Concatenate()([ts_out, s])
    h = layers.Dense(64, activation='relu')(h)
    h = layers.Dropout(0.25)(h)
    out = layers.Dense(n_classes, activation='softmax')(h)
    
    model = Model(inputs=[ts_in, st_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_confusion_matrix(cm, classes, temp_dir, normalize=False):
    """Plot and save confusion matrix"""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    filename = f'confusion_matrix{"_norm" if normalize else ""}.png'
    filepath = os.path.join(temp_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    return filepath


def train_model(df, temp_dir):
    """Train the CNN+GRU model"""
    print("\nTraining model...")
    
    # Build time-series tensor
    X_ts, used_cols = build_time_series_tensor(df, ModelConfig.SEQ_LEN)
    print(f"Time-series shape: {X_ts.shape}")
    
    # Extract static features
    static_cols = ['BMI', 'Age', 'activity_type_encoded']
    X_static = df[static_cols].copy()
    for c in static_cols:
        X_static[c] = pd.to_numeric(X_static[c], errors='coerce')
    X_static = X_static.fillna(X_static.median())
    
    scaler = StandardScaler()
    X_static = scaler.fit_transform(X_static.values)
    
    # Standardize time-series
    X_ts = standardize_channels(X_ts)
    
    # Labels
    y = pd.to_numeric(df['label_encoded'], errors='coerce').fillna(0).astype(int).values
    
    # Split data
    Xts_train, Xts_temp, Xst_train, Xst_temp, y_train, y_temp = train_test_split(
        X_ts, X_static, y, test_size=0.30, random_state=42, stratify=y
    )
    Xts_val, Xts_test, Xst_val, Xst_test, y_val, y_test = train_test_split(
        Xts_temp, Xst_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    
    # Build model
    model = build_model(
        seq_len=ModelConfig.SEQ_LEN,
        n_channels=X_ts.shape[2],
        static_dim=X_static.shape[1],
        n_classes=3
    )
    
    model.summary()
    
    # Model path
    model_path = os.path.join(temp_dir, 'final_model.keras')
    
    # Callbacks
    cb = [
        callbacks.EarlyStopping(monitor='val_loss', patience=8, 
                               restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=4, min_lr=1e-5, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor='val_loss', 
                                 save_best_only=True)
    ]
    
    # Train
    start_time = time.time()
    history = model.fit(
        [Xts_train, Xst_train], y_train,
        validation_data=([Xts_val, Xst_val], y_val),
        epochs=ModelConfig.EPOCHS,
        batch_size=ModelConfig.BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Evaluate
    y_prob = model.predict([Xts_test, Xst_test], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Bad", "Moderate", "Healthy"],
        zero_division=0
    ))
    
    # Confusion matrices
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    cm_path = plot_confusion_matrix(cm, ["Bad", "Moderate", "Healthy"], 
                                    temp_dir, normalize=False)
    cm_norm_path = plot_confusion_matrix(cm, ["Bad", "Moderate", "Healthy"], 
                                         temp_dir, normalize=True)
    
    results = {
        'model_path': model_path,
        'cm_path': cm_path,
        'cm_norm_path': cm_norm_path,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }
    
    return model, results


# =====================================================
# MAIN PIPELINE
# =====================================================
def run_aws_pipeline():
    """Execute complete pipeline with AWS integration"""
    print("\n" + "="*70)
    print("AWS CLOUD KNEE HEALTH CLASSIFICATION PIPELINE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    try:
        # Initialize S3 handler
        s3 = S3Handler()
        
        # Create temporary directory for local operations
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\nTemporary directory: {temp_dir}")
            
            # STEP 1: Download data from S3
            print(f"\n{'='*60}")
            print("STEP 1: DOWNLOADING DATA FROM S3")
            print(f"{'='*60}")
            
            climbing_df = s3.download_csv_from_s3(
                AWSConfig.CLIMBING_FILE, 
                AWSConfig.S3_INPUT_PREFIX
            )
            walking_df = s3.download_csv_from_s3(
                AWSConfig.WALKING_FILE, 
                AWSConfig.S3_INPUT_PREFIX
            )
            
            # STEP 2: Clean data
            print(f"\n{'='*60}")
            print("STEP 2: CLEANING DATA")
            print(f"{'='*60}")
            climbing_df = clean_dataset(climbing_df)
            walking_df = clean_dataset(walking_df)
            
            # STEP 3: Transform data
            print(f"\n{'='*60}")
            print("STEP 3: TRANSFORMING DATA")
            print(f"{'='*60}")
            climbing_df = transform_to_sensor_columns(climbing_df)
            walking_df = transform_to_sensor_columns(walking_df)
            
            # STEP 4: Combine datasets
            print(f"\n{'='*60}")
            print("STEP 4: COMBINING DATASETS")
            print(f"{'='*60}")
            combined_df = combine_datasets(climbing_df, walking_df)
            
            # Upload combined dataset to S3
            s3.upload_csv_to_s3(
                combined_df, 
                'combined_dataset.csv', 
                AWSConfig.S3_OUTPUT_PREFIX
            )
            
            # STEP 5: Encode labels
            print(f"\n{'='*60}")
            print("STEP 5: ENCODING LABELS")
            print(f"{'='*60}")
            encoded_df, label_enc, activity_enc = encode_labels(combined_df)
            
            # Upload encoded dataset to S3
            s3.upload_csv_to_s3(
                encoded_df, 
                'encoded_dataset.csv', 
                AWSConfig.S3_OUTPUT_PREFIX
            )
            
            # STEP 6: Train model
            print(f"\n{'='*60}")
            print("STEP 6: TRAINING MODEL")
            print(f"{'='*60}")
            model, results = train_model(encoded_df, temp_dir)
            
            # STEP 7: Upload results to S3
            print(f"\n{'='*60}")
            print("STEP 7: UPLOADING RESULTS TO S3")
            print(f"{'='*60}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Upload model
            s3.upload_model_to_s3(
                results['model_path'],
                f'knee_health_model_{timestamp}.keras'
            )
            
            # Upload confusion matrices
            s3.upload_file_to_s3(
                results['cm_path'],
                f'confusion_matrix_{timestamp}.png',
                AWSConfig.S3_OUTPUT_PREFIX
            )
            s3.upload_file_to_s3(
                results['cm_norm_path'],
                f'confusion_matrix_norm_{timestamp}.png',
                AWSConfig.S3_OUTPUT_PREFIX
            )
            
            # Save and upload metrics
            metrics_df = pd.DataFrame([{
                'timestamp': timestamp,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score']
            }])
            
            metrics_path = os.path.join(temp_dir, 'metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            s3.upload_file_to_s3(
                metrics_path,
                f'metrics_{timestamp}.csv',
                AWSConfig.S3_OUTPUT_PREFIX
            )
        
        pipeline_time = time.time() - pipeline_start
        
        print(f"\n{'='*70}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        print(f"Total time: {pipeline_time:.2f} seconds ({pipeline_time/60:.2f} minutes)")
        print(f"\nAll outputs saved to S3:")
        print(f"  Bucket: {AWSConfig.S3_BUCKET_NAME}")
        print(f"  Model: {AWSConfig.S3_MODEL_PREFIX}knee_health_model_{timestamp}.keras")
        print(f"  Metrics: {AWSConfig.S3_OUTPUT_PREFIX}metrics_{timestamp}.csv")
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed!")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    print("\n" + "="*70)
    print("SETUP INSTRUCTIONS:")
    print("="*70)
    print("1. Set AWS credentials as environment variables:")
    print("   export AWS_ACCESS_KEY_ID='your_access_key'")
    print("   export AWS_SECRET_ACCESS_KEY='your_secret_key'")
    print("   export AWS_REGION='us-east-1'")
    print("\n2. Update AWSConfig class with your S3 bucket name")
    print("\n3. Ensure your S3 bucket has:")
    print("   - input_data/Climbing_Data_2.csv")
    print("   - input_data/Walking_Data_2.csv")
    print("="*70 + "\n")
    
    # Run pipeline
    success = run_aws_pipeline()
    
    if success:
        print("\nModel is ready for deployment!")
    else:
        print("\nPipeline execution failed.")