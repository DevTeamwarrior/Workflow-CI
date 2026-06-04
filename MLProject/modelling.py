import os
import sys
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Rescaling, GlobalAveragePooling2D, Conv2D, BatchNormalization
from tensorflow.keras.utils import image_dataset_from_directory
import mlflow

# Mengambil parameter input dari eksekusi MLProject
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=15)
args = parser.parse_args()

print(f"=== RE-TRAINING BERJALAN: LR={args.learning_rate}, EPOCHS={args.epochs} ===")

# Pengaturan direktori (Ganti sesuai kebutuhan environment/LFS Anda)
PREPROCESSED_DIR = 'namadataset_preprocessing'

if os.path.exists(PREPROCESSED_DIR):
    train_ds = image_dataset_from_directory(os.path.join(PREPROCESSED_DIR, 'train'), image_size=(150, 150), batch_size=32, label_mode='categorical')
    val_ds = image_dataset_from_directory(os.path.join(PREPROCESSED_DIR, 'val'), image_size=(150, 150), batch_size=32, label_mode='categorical')
    num_classes = len(train_ds.class_names)
else:
    print("Dataset dummy/not found untuk mode build container.")
    num_classes = 5

# Arsitektur Model (Sama seperti Kriteria 2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    Rescaling(1./127.5, offset=-1, input_shape=(150, 150, 3)),
    base_model,
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Hubungkan eksekusi ke DagsHub/MLflow
with mlflow.start_run():
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("epochs", args.epochs)
    
    if os.path.exists(PREPROCESSED_DIR):
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
        mlflow.log_metric("final_accuracy", history.history['accuracy'][-1])
    
    mlflow.keras.log_model(model, artifact_path="model")
print("=== RE-TRAINING SELESAI ===")