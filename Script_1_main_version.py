# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import random

# To deal witht the Directions and folders
import os
from pathlib import Path
import requests

# set the random seed
# the second thing we should set the random seed with 42 number
random.seed(42)

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd

# Function to create a DataFrame from image directories
def create_dataframe(base_path):
    data = []
    classes = ['autistic', 'non_autistic']

    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            data.append((img_path, 1 if class_name == 'autistic' else 0))

    return pd.DataFrame(data, columns=['filename', 'label'])

# Create DataFrames for train, validation, and test datasets
train_df = create_dataframe(train_path)
test_df = create_dataframe(test_path)

# Display the first few rows of the DataFrame
print(train_df.head())

# Function to create a DataFrame from image directories
def create_dataframe(base_path):
    data = []
    classes = ['Autistic', 'Non_Autistic']

    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            data.append((img_path, 1 if class_name == 'Autistic' else 0))

    return pd.DataFrame(data, columns=['filename', 'label'])

# Create DataFrames for train, validation, and test datasets
val_df = create_dataframe(val_path)

# Display the first few rows of the DataFrame
print(val_df.head())

# Convert label column to string
train_df['label'] = train_df['label'].astype(str)
val_df['label'] = val_df['label'].astype(str)
test_df['label'] = test_df['label'].astype(str)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image data generator
train_data_gen = ImageDataGenerator(rescale=1.0 / 255)
val_data_gen = ImageDataGenerator(rescale=1.0 / 255)
test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Train generator
train_gen = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='label',
    target_size=(256, 256),
    class_mode='binary',
    batch_size=32,
    shuffle=True
)

# Validation generator
val_gen = val_data_gen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='label',
    target_size=(256, 256),
    class_mode='binary',
    batch_size=32,
    shuffle=True
)

# Test generator
test_gen = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col='label',
    target_size=(256, 256),
    class_mode='binary',
    batch_size=32,
    shuffle=False
)

import tensorflow as tf

# Squeeze-and-Excitation block (Attention Mechanism)
def se_block(input_tensor, reduction_ratio=16):
    """ Squeeze-and-Excitation block """
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // reduction_ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    x = tf.keras.layers.multiply([input_tensor, se])
    return x

# Custom CNN Backbone
def custom_cnn_backbone(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 128x128

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 64x64

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 32x32

    return inputs, x

# Input shape
input_shape = (256, 256, 3)

# Build the model
inputs, feature_maps = custom_cnn_backbone(input_shape)

# Apply Squeeze-and-Excitation block on the final feature maps
feature_maps_se = se_block(feature_maps)

print("Feature Maps shape after SE block:", feature_maps_se.shape)

# Flatten the feature maps
x = tf.keras.layers.Flatten()(feature_maps_se)

# Dense layer
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Output layer (Binary classification)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Final model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks: Early stopping, model checkpointing, and learning rate reduction
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
mc = ModelCheckpoint(filepath='best_model_attention.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')

callbacks = [es, mc, rl]

from sklearn.utils import class_weight
import numpy as np


class_names = ['1', '0']



# Compute class weights
train_labels = train_gen.labels
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',  # 'balanced' automatically adjusts for class imbalance
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert the class_weights array into a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Now, train the model with class_weight
# epochs = 20
# history_2 = model.fit(
#     train_gen,
#     steps_per_epoch=188,
#     validation_data=val_gen,
#     validation_steps=20,
#     epochs=epochs,
#     class_weight=class_weight_dict,  # Add this argument for class balancing
#     callbacks=callbacks
# )

# Plot accuracy curve
history = history_2
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_title('SE-CNN model Accuracy', fontsize=16)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
plt.show()

# Plot loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('SE-CNN model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.show()

import numpy as np

# Get the true labels from the test generator
test_labels = test_gen.classes

# Make predictions on the test set
predictions = model.predict(test_gen)
predicted_classes = np.round(predictions).astype(int).flatten()

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix


# Calculate accuracy
test_accuracy = accuracy_score(test_labels, predicted_classes)

# Calculate precision, recall, F1-score, and Cohen's Kappa
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)
kappa = cohen_kappa_score(test_labels, predicted_classes)

# Calculate confusion matrix for specificity
cm = confusion_matrix(test_labels, predicted_classes)
TN, FP, FN, TP = cm.ravel()

# Calculate specificity
specificity = TN / (TN + FP)
avg_specificity = specificity

# Print the results
print("Testing Accuracy:", test_accuracy)
print("Precision (PPV):", precision)
print("Recall (Sensitivity):", recall)
print("F1-Score:", f1)
print("Cohen's Kappa Score:", kappa)
print("Average Specificity:", avg_specificity)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

classes = {0: 'Non-Autistic', 1: 'Autistic'}

# Compute the confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(8, 8))

# Generate the heatmap with Seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=list(classes.values()), yticklabels=list(classes.values()), ax=ax, linewidths=0.5, linecolor='gray')

# Set the axis labels and title
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title('Confusion Matrix for SE-CNN model', fontsize=16)

# Set the colorbar properties
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Set the tick labels' font size
ax.tick_params(axis='both', labelsize=12)

# Display the figure
plt.show()

# Generate classification report
from sklearn.metrics import classification_report

report = classification_report(test_labels, predicted_classes)

print(report);

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the test set
predicted_probs = model.predict(test_gen)  # Predicted probabilities

# Calculate the FPR and TPR
fpr, tpr, thresholds = roc_curve(test_labels, predicted_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SE-CNN model')
plt.legend(loc='lower right')
plt.grid()
plt.show()

import tensorflow as tf

# Squeeze-and-Excitation block (Attention Mechanism)
def se_block(input_tensor, reduction_ratio=16):
    """ Squeeze-and-Excitation block """
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // reduction_ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    x = tf.keras.layers.multiply([input_tensor, se])
    return x

# Custom CNN Backbone
def custom_cnn_backbone(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 128x128

    # Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 64x64

    # Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)  # 32x32

    # Extract multi-scale features
    layer1 = x  # Features from Block 1 (32 filters)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    layer2 = x  # Features from Block 2 (64 filters)

    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    layer3 = x  # Features from Block 3 (128 filters)

    return inputs, layer1, layer2, layer3

# Input shape
input_shape = (256, 256, 3)

# Build the model
inputs, layer1, layer2, layer3 = custom_cnn_backbone(input_shape)

# Apply Squeeze-and-Excitation block
layer1_se = se_block(layer1)
layer2_se = se_block(layer2)
layer3_se = se_block(layer3)

print("Layer 1 shape:", layer1_se.shape)
print("Layer 2 shape:", layer2_se.shape)
print("Layer 3 shape:", layer3_se.shape)

# Up-sampling all feature maps to a common size
layer1_upsampled = tf.keras.layers.Conv2D(64, (1, 1), activation='relu')(tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(layer1_se))  # Resize to 16x16
layer2_upsampled = tf.keras.layers.Conv2D(128, (1, 1), activation='relu')(layer2_se)  # Already 16x16
layer3_upsampled = tf.keras.layers.Conv2D(256, (1, 1), activation='relu')(tf.keras.layers.UpSampling2D(size=(2, 2))(layer3_se))  # Resize to 16x16

# Multi-scale feature fusion via concatenation
fused_features = tf.keras.layers.Concatenate()([layer1_se, layer2_se, layer3_se])

# Flatten the fused features
x = tf.keras.layers.Flatten()(fused_features)

# Dense layer
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Output layer (Binary classification)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Final model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Callbacks: Early stopping, model checkpointing, and learning rate reduction
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
mc = ModelCheckpoint(filepath='best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')

callbacks = [es, mc, rl]

from sklearn.utils import class_weight
import numpy as np


class_names = ['1', '0']



# Compute class weights
train_labels = train_gen.labels
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',  # 'balanced' automatically adjusts for class imbalance
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert the class_weights array into a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Now, train the model with class_weight
# epochs = 20
# history_2 = model.fit(
#     train_gen,
#     steps_per_epoch=188,
#     validation_data=val_gen,
#     validation_steps=20,
#     epochs=epochs,
#     class_weight=class_weight_dict,
#     callbacks=callbacks
# )

# Plot accuracy curve
history = history_2
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Training Accuracy')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax.set_title('SE-MSF-CNN model Accuracy', fontsize=16)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.legend(loc='lower right', fontsize=12)
ax.tick_params(axis='both', labelsize=12)
plt.show()

# Plot loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('SE-MSF-CNN model Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.show()

import numpy as np

# Get the true labels from the test generator
test_labels = test_gen.classes

# Make predictions on the test set
predictions = model.predict(test_gen)
predicted_classes = np.round(predictions).astype(int).flatten()  # Convert probabilities to binary predictions

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix


# Calculate accuracy
train_accuracy = accuracy_score(train_labels, model.predict(train_gen).round())
test_accuracy = accuracy_score(test_labels, predicted_classes)

# Calculate precision, recall, F1-score, and Cohen's Kappa
precision = precision_score(test_labels, predicted_classes)
recall = recall_score(test_labels, predicted_classes)
f1 = f1_score(test_labels, predicted_classes)
kappa = cohen_kappa_score(test_labels, predicted_classes)

# Calculate confusion matrix for specificity
cm = confusion_matrix(test_labels, predicted_classes)
TN, FP, FN, TP = cm.ravel()

# Calculate specificity
specificity = TN / (TN + FP)
avg_specificity = specificity

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Precision (PPV):", precision)
print("Recall (Sensitivity):", recall)
print("F1-Score:", f1)
print("Cohen's Kappa Score:", kappa)
print("Average Specificity:", avg_specificity)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

classes = {0: 'Non-Autistic', 1: 'Autistic'}

# Compute the confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)

# Create a figure with a single subplot
fig, ax = plt.subplots(figsize=(8, 8))

# Generate the heatmap with Seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=list(classes.values()), yticklabels=list(classes.values()), ax=ax, linewidths=0.5, linecolor='gray')

# Set the axis labels and title
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title('Confusion Matrix for SE-MSF-CNN model', fontsize=16)

# Set the colorbar properties
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Set the tick labels' font size
ax.tick_params(axis='both', labelsize=12)

# Display the figure
plt.show()

# Generate classification report
from sklearn.metrics import classification_report

report_sec = classification_report(test_labels, predicted_classes)

print(report_sec);

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the test set
predicted_probs = model.predict(test_gen)

# Calculate the FPR and TPR
fpr, tpr, thresholds = roc_curve(test_labels, predicted_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SE-MSF-CNN model')
plt.legend(loc='lower right')
plt.grid()
plt.show()
