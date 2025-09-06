# =====================================
# 3. Clustering and Model Setup (FIXED VERSION)
# =====================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Clear any previous cluster labels to avoid overwrites
if 'cluster_labels' in locals():
    del cluster_labels

# Recompute KMeans clustering on resampled data
X_flat_resampled = X_spectrograms_resampled.reshape(X_spectrograms_resampled.shape[0], -1)  # Flatten resampled data
scaler_resampled = StandardScaler()
X_scaled_resampled = scaler_resampled.fit_transform(X_flat_resampled)  # Scale resampled data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels_resampled = kmeans.fit_predict(X_scaled_resampled)  # Compute new cluster labels

# Verify cluster labels range and count
max_cluster_idx = np.max(cluster_labels_resampled)
print(f"Maximum cluster index: {max_cluster_idx}")
print(f"Number of samples in cluster_labels_resampled: {len(cluster_labels_resampled)}")

# Convert to categorical
cluster_labels_categorical_resampled = tf.keras.utils.to_categorical(cluster_labels_resampled, num_classes=n_clusters)

# Debug: Verify shapes
print(f"X_spectrograms_resampled shape: {X_spectrograms_resampled.shape}")
print(f"Y_categorical shape: {Y_categorical.shape}")
print(f"cluster_labels_categorical_resampled shape: {cluster_labels_categorical_resampled.shape}")

# Build Clustered CNN model with improved regularization
def build_cnn_model_with_clustering(input_shape, num_classes, n_clusters):
    inputs = layers.Input(shape=input_shape)
    
    # First conv block with stronger regularization
    x = layers.Conv2D(32, (3, 3), activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Increased dropout
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Increased dropout
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)  # Increased dropout
    
    # Fourth conv block for better feature extraction
    x = layers.Conv2D(256, (3, 3), activation='relu', 
                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)  # Use GlobalAveragePooling instead of Flatten
    x = layers.Dropout(0.4)(x)
    
    # Dense layers with regularization
    clustering_features = layers.Dense(512, activation='relu', 
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    clustering_features = layers.Dropout(0.5)(clustering_features)
    
    clustering_features = layers.Dense(256, activation='relu', 
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01))(clustering_features)
    clustering_features = layers.Dropout(0.5)(clustering_features)
    
    # Output layers
    classification_output = layers.Dense(num_classes, activation='softmax', 
                                        name='classification_output')(clustering_features)
    clustering_output = layers.Dense(n_clusters, activation='softmax', 
                                    name='clustering_output')(clustering_features)
    
    return models.Model(inputs=inputs, outputs=[classification_output, clustering_output])

# Compile and train the model with improved settings
model = build_cnn_model_with_clustering(input_shape, len(class_names), n_clusters)

# Use a lower learning rate and add learning rate scheduling
initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss={'classification_output': 'categorical_crossentropy', 'clustering_output': 'categorical_crossentropy'},
    loss_weights={'classification_output': 1.0, 'clustering_output': cluster_weight},
    metrics={'classification_output': ['accuracy'], 'clustering_output': ['accuracy']}
)

# Split data for training and testing
X_train, X_test, Y_train, Y_test, cluster_train, cluster_test = train_test_split(
    X_spectrograms_resampled, Y_categorical, cluster_labels_categorical_resampled, 
    test_size=0.2, random_state=42, stratify=np.argmax(Y_categorical, axis=1)
)

# Custom generator without augmentation
def multi_output_generator(x, y_class, y_cluster, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y_class, y_cluster))
    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y_class, y_cluster: (x, {'classification_output': y_class, 'clustering_output': y_cluster}))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Train the model with improved callbacks
train_generator = multi_output_generator(X_train, Y_train, cluster_train, batch_size=32)

# Add more callbacks to prevent overfitting
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_classification_output_loss',
        mode='min',
        patience=10,  # Increased patience
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_classification_output_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_classification_output_accuracy',
        save_best_only=True,
        mode='max'
    )
]

history = model.fit(
    train_generator,
    validation_data=(X_test, {'classification_output': Y_test, 'clustering_output': cluster_test}),
    epochs=100,  # Increased epochs since we have early stopping
    steps_per_epoch=len(X_train) // 32,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on training set
train_loss, train_classification_loss, train_clustering_loss, train_classification_acc, train_clustering_acc = model.evaluate(
    X_train, {'classification_output': Y_train, 'clustering_output': cluster_train}, verbose=0
)
print(f"Training Accuracy: {train_classification_acc:.4f}")

# Evaluate on test set
test_loss, test_classification_loss, test_clustering_loss, test_classification_acc, test_clustering_acc = model.evaluate(
    X_test, {'classification_output': Y_test, 'clustering_output': cluster_test}, verbose=0
)
print(f"Test Accuracy: {test_classification_acc:.4f}")

# =====================================
# FIXED METRICS CALCULATION FOR MULTICLASS
# =====================================

from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# Get predictions
y_pred = model.predict(X_test)
y_pred_class = y_pred[0]  # Classification output probabilities
y_true_class = np.argmax(Y_test, axis=1)

# For multiclass, we need to handle it differently
num_classes = len(class_names)

# Binarize the labels for multiclass ROC
y_test_bin = label_binarize(y_true_class, classes=range(num_classes))

# Calculate ROC AUC for each class and average
roc_auc_scores = []
for i in range(num_classes):
    if len(np.unique(y_test_bin[:, i])) > 1:  # Check if class exists in test set
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_class[:, i])
        roc_auc = auc(fpr, tpr)
        roc_auc_scores.append(roc_auc)
        print(f"ROC AUC for class {class_names[i]}: {roc_auc:.4f}")

# Calculate macro-averaged ROC AUC
macro_roc_auc = np.mean(roc_auc_scores)
print(f"Macro-averaged ROC AUC: {macro_roc_auc:.4f}")

# Calculate micro-averaged ROC AUC
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_pred_class.ravel())
micro_roc_auc = auc(fpr_micro, tpr_micro)
print(f"Micro-averaged ROC AUC: {micro_roc_auc:.4f}")

# Precision-Recall curves for each class
precision_recall_aucs = []
for i in range(num_classes):
    if len(np.unique(y_test_bin[:, i])) > 1:  # Check if class exists in test set
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_class[:, i])
        pr_auc = auc(recall, precision)
        precision_recall_aucs.append(pr_auc)
        print(f"Precision-Recall AUC for class {class_names[i]}: {pr_auc:.4f}")

# Calculate macro-averaged Precision-Recall AUC
macro_pr_auc = np.mean(precision_recall_aucs)
print(f"Macro-averaged Precision-Recall AUC: {macro_pr_auc:.4f}")

# Additional metrics
from sklearn.metrics import classification_report, confusion_matrix

# Classification report
print("\nClassification Report:")
print(classification_report(y_true_class, np.argmax(y_pred_class, axis=1), target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true_class, np.argmax(y_pred_class, axis=1))
print("\nConfusion Matrix:")
print(cm)

# Save history for curves
history_dict = history.history
print("\nTraining History Keys:", list(history_dict.keys()))

# Plot training curves
plt.figure(figsize=(15, 5))

# Plot training & validation accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['classification_output_accuracy'])
plt.plot(history.history['val_classification_output_accuracy'])
plt.title('Model Classification Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(1, 3, 2)
plt.plot(history.history['classification_output_loss'])
plt.plot(history.history['val_classification_output_loss'])
plt.title('Model Classification Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot clustering accuracy
plt.subplot(1, 3, 3)
plt.plot(history.history['clustering_output_accuracy'])
plt.plot(history.history['val_clustering_output_accuracy'])
plt.title('Model Clustering Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

print(f"\nFinal Results:")
print(f"Training Accuracy: {train_classification_acc:.4f}")
print(f"Test Accuracy: {test_classification_acc:.4f}")
print(f"Overfitting Gap: {train_classification_acc - test_classification_acc:.4f}")
print(f"Macro-averaged ROC AUC: {macro_roc_auc:.4f}")
print(f"Macro-averaged Precision-Recall AUC: {macro_pr_auc:.4f}")