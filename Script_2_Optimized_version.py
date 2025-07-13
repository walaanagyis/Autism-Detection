# -*- coding: utf-8 -*-

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

    # Remove multi-scale feature extraction
    # You can use the output of Block 3 directly
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
mc = ModelCheckpoint(filepath='optimized_best_model_attention_first_arch.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')

callbacks = [es, mc, rl]

from sklearn.utils import class_weight
import numpy as np

class_names = ['1', '0']



# Compute class weights
train_labels = train_gen.labels
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert the class_weights array into a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Custom training callback
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

def objective_function(params):
    learning_rate, num_neurons_layer1, num_neurons_layer2 = params

    try:
        # Train the model
        history = model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            validation_data=val_gen,
            validation_steps=len(val_gen),
            epochs=1,
            callbacks=[TrainingProgressCallback()],
            batch_size=32,
            verbose=1
        )

        # Evaluate the model on validation data
        val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    except Exception as e:
        print(f"Error during model training/evaluation: {e}")
        val_loss, val_acc = float('inf'), 0.0  # Assign default values in case of error

    # Return accuracy and loss for the optimizer
    return val_acc, val_loss

def initial_population(num_sharks, dim, ub, lb):
    return np.random.rand(num_sharks, dim) * (ub - lb) + lb

# Define the plotting function

def plot_optimization_results(acc_curve, loss_curve, ccurve):
    """Plot accuracy, loss, and convergence"""

    # Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(acc_curve, label='Accuracy')
    plt.title('WSO-SE-CNN Optimization Results - Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label='Loss', color='orange')
    plt.title('WSO-SE-CNN Optimization Results - Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(ccurve, label='Convergence', color='green')
    plt.title('WSO-SE-CNN Optimization Results - Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Best Score (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.show()

# White Shark Optimization (WSO) algorithm
def WSO(num_sharks, max_iter, lb, ub, dim, fobj):
    ccurve = np.zeros(max_iter)  # Convergence curve
    acc_curve = np.zeros(max_iter)  # Accuracy curve
    loss_curve = np.zeros(max_iter)  # Loss curve

    WSO_Positions = initial_population(num_sharks, dim, ub, lb)
    v = np.zeros_like(WSO_Positions)

    # Initialize arrays to store accuracies and losses
    fitness_acc_loss = [fobj(WSO_Positions[i, :]) for i in range(num_sharks)]
    fitness = np.array([f[0] for f in fitness_acc_loss])  # Accuracy
    losses = np.array([f[1] for f in fitness_acc_loss])   # Loss

    # Initialize variables
    fmax0 = np.max(fitness)  # Best accuracy
    index = np.argmax(fitness)
    wbest = WSO_Positions.copy()
    gbest = WSO_Positions[index, :].copy()

    fmax = 0.75
    fmin = 0.07
    tau = 4.11
    mu = 2 / abs(2 - tau - np.sqrt(tau**2 - 4 * tau))
    pmin = 0.5
    pmax = 1.5
    a0 = 6.25
    a1 = 100
    a2 = 0.0005

    for ite in range(max_iter):
        mv = 1 / (a0 + np.exp((max_iter / 2.0 - ite) / a1))
        s_s = abs((1 - np.exp(-a2 * ite / max_iter)))
        p1 = pmax + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)
        p2 = pmin + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)
        nu = np.random.randint(0, num_sharks, num_sharks)
        for i in range(num_sharks):
            rr = 1 + np.random.rand() * 2
            wr = abs(((2 * np.random.rand()) - (1 * np.random.rand() + np.random.rand())) / rr)
            v[i, :] = mu * v[i, :] + wr * (wbest[nu[i], :] - WSO_Positions[i, :])

        for i in range(num_sharks):
            f = fmin + (fmax - fmin) / (fmax + fmin)
            a = np.sign(WSO_Positions[i, :] - ub) > 0
            b = np.sign(WSO_Positions[i, :] - lb) < 0
            wo = np.logical_xor(a, b)
            if np.random.rand() < mv:
                WSO_Positions[i, :] = WSO_Positions[i, :] * (~wo) + (ub * a + lb * b)
            else:
                WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f

        for i in range(num_sharks):
            for j in range(dim):
                if np.random.rand() < s_s:
                    Dist = abs(np.random.rand() * (gbest[j] - WSO_Positions[i, j]))
                    if i == 1:
                        WSO_Positions[i, j] = gbest[j] + np.random.rand() * Dist * np.sign(np.random.rand() - 0.5)
                    else:
                        WSO_Pos = gbest[j] + np.random.rand() * Dist * np.sign(np.random.rand() - 0.5)
                        WSO_Positions[i, j] = (WSO_Pos + WSO_Positions[i - 1, j]) / 2 * np.random.rand()

        # Update fitness and losses
        for i in range(num_sharks):
            if np.all(WSO_Positions[i, :] >= lb) and np.all(WSO_Positions[i, :] <= ub):
                fit_acc_loss = fobj(WSO_Positions[i, :])
                fitness[i], losses[i] = fit_acc_loss  # Accuracy and loss

                if fitness[i] > np.max(fitness):  # Maximize accuracy
                    wbest[i, :] = WSO_Positions[i, :].copy()

                if fitness[i] > fmax0:  # Maximize accuracy
                    fmax0 = fitness[i]
                    gbest = wbest[i, :].copy()

        # Store accuracy, loss, and convergence data for plotting
        acc_curve[ite] = np.max(fitness)
        loss_curve[ite] = np.min(losses)
        ccurve[ite] = fmax0

        if ite > 1:
            plt.plot([ite - 1, ite], [ccurve[ite - 1], ccurve[ite]], 'b')
            plt.title('Convergence characteristic curve')
            plt.xlabel('Iteration')
            plt.ylabel('Best score obtained so far')
            plt.draw()
            plt.pause(0.01)

    # Plot optimization results
    plot_optimization_results(acc_curve, loss_curve, ccurve)

    return fmax0, gbest, ccurve

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Set Hyperparameter Bounds and Run WSO
# lb = np.array([0.0001, 10, 10])
# ub = np.array([0.1, 200, 200])
# dim = 3
# num_sharks = 10
# max_iter = 5
# 
# fmax0, gbest, ccurve = WSO(num_sharks, max_iter, lb, ub, dim, objective_function)
# print("Best Hyperparameters:", gbest)
# print("Best Accuracy:", fmax0)

# Get the true labels from the test generator
test_labels = test_gen.classes

# Make predictions on the test set
predictions = model_load.predict(test_gen)
predicted_classes = np.round(predictions).astype(int).flatten()  # Convert probabilities to binary predictions

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
fig, ax = plt.subplots(figsize=(5, 5))

# Generate the heatmap with Seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=list(classes.values()), yticklabels=list(classes.values()), ax=ax, linewidths=0.5, linecolor='gray')

# Set the axis labels and title
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title('Confusion Matrix for WSO-SE-CNN model', fontsize=16)

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
predicted_probs = model_load.predict(test_gen)  # Predicted probabilities

# Calculate the FPR and TPR
fpr, tpr, thresholds = roc_curve(test_labels, predicted_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for WSO-SE-CNN model')
plt.legend(loc='lower right')
plt.grid()
plt.show()

from sklearn.metrics import precision_recall_curve, auc

# Get true labels from the test generator
true_labels = test_gen.classes

# Get predicted probabilities for the test set
predicted_probs = model_load.predict(test_gen)  # Predicted probabilities

# Flatten the predicted probabilities to 1D array
predicted_probs_flat = predicted_probs.flatten()

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(true_labels, predicted_probs_flat)

# Calculate AUC for PR curve
pr_auc = auc(recall, precision)

# Plotting the Precision-Recall curve
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve (area = {:.2f})'.format(pr_auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for WSO-SE-CNN model')
plt.legend(loc='lower left')
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

from sklearn.utils import class_weight
import numpy as np

class_names = ['1', '0']



# Compute class weights
train_labels = train_gen.labels
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)


# Convert the class_weights array into a dictionary
class_weight_dict = dict(enumerate(class_weights))

# Callbacks: Early stopping, model checkpointing, and learning rate reduction
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
mc = ModelCheckpoint(filepath='optimized_best_model_final.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='min')

callbacks = [es, mc, rl]

# Custom training callback
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val Loss: {logs['val_loss']:.4f}, Val Accuracy: {logs['val_accuracy']:.4f}")

def objective_function(params):
    learning_rate, num_neurons_layer1, num_neurons_layer2 = params

    try:
        # Train the model
        history = model.fit(
            train_gen,
            steps_per_epoch=len(train_gen),
            validation_data=val_gen,
            validation_steps=len(val_gen),
            epochs=1,
            callbacks=[TrainingProgressCallback()],
            batch_size=32,
            verbose=1
        )

        # Evaluate the model on validation data
        val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    except Exception as e:
        print(f"Error during model training/evaluation: {e}")
        val_loss, val_acc = float('inf'), 0.0

    # Return accuracy and loss for the optimizer
    return val_acc, val_loss

def initial_population(num_sharks, dim, ub, lb):
    return np.random.rand(num_sharks, dim) * (ub - lb) + lb

# Define the plotting function

def plot_optimization_results(acc_curve, loss_curve, ccurve):
    """Plot accuracy, loss, and convergence"""

    # Plot Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(acc_curve, label='Accuracy')
    plt.title('WSO-SE-MSF-CNN Optimization Results - Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_curve, label='Loss', color='orange')
    plt.title('WSO-SE-MSF-CNN Optimization Results - Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(ccurve, label='Convergence', color='green')
    plt.title('WSO-SE-MSF-CNN Optimization Results - Convergence')
    plt.xlabel('Iterations')
    plt.ylabel('Best Score (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.show()

# White Shark Optimization (WSO) algorithm
def WSO(num_sharks, max_iter, lb, ub, dim, fobj):
    ccurve = np.zeros(max_iter)  # Convergence curve
    acc_curve = np.zeros(max_iter)  # Accuracy curve
    loss_curve = np.zeros(max_iter)  # Loss curve

    WSO_Positions = initial_population(num_sharks, dim, ub, lb)
    v = np.zeros_like(WSO_Positions)

    # Initialize arrays to store accuracies and losses
    fitness_acc_loss = [fobj(WSO_Positions[i, :]) for i in range(num_sharks)]
    fitness = np.array([f[0] for f in fitness_acc_loss])  # Accuracy
    losses = np.array([f[1] for f in fitness_acc_loss])   # Loss

    # Initialize variables
    fmax0 = np.max(fitness)  # Best accuracy
    index = np.argmax(fitness)
    wbest = WSO_Positions.copy()
    gbest = WSO_Positions[index, :].copy()

    fmax = 0.75
    fmin = 0.07
    tau = 4.11
    mu = 2 / abs(2 - tau - np.sqrt(tau**2 - 4 * tau))
    pmin = 0.5
    pmax = 1.5
    a0 = 6.25
    a1 = 100
    a2 = 0.0005

    for ite in range(max_iter):
        mv = 1 / (a0 + np.exp((max_iter / 2.0 - ite) / a1))
        s_s = abs((1 - np.exp(-a2 * ite / max_iter)))
        p1 = pmax + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)
        p2 = pmin + (pmax - pmin) * np.exp(-(4 * ite / max_iter) ** 2)
        nu = np.random.randint(0, num_sharks, num_sharks)
        for i in range(num_sharks):
            rr = 1 + np.random.rand() * 2
            wr = abs(((2 * np.random.rand()) - (1 * np.random.rand() + np.random.rand())) / rr)
            v[i, :] = mu * v[i, :] + wr * (wbest[nu[i], :] - WSO_Positions[i, :])

        for i in range(num_sharks):
            f = fmin + (fmax - fmin) / (fmax + fmin)
            a = np.sign(WSO_Positions[i, :] - ub) > 0
            b = np.sign(WSO_Positions[i, :] - lb) < 0
            wo = np.logical_xor(a, b)
            if np.random.rand() < mv:
                WSO_Positions[i, :] = WSO_Positions[i, :] * (~wo) + (ub * a + lb * b)
            else:
                WSO_Positions[i, :] = WSO_Positions[i, :] + v[i, :] / f

        for i in range(num_sharks):
            for j in range(dim):
                if np.random.rand() < s_s:
                    Dist = abs(np.random.rand() * (gbest[j] - WSO_Positions[i, j]))
                    if i == 1:
                        WSO_Positions[i, j] = gbest[j] + np.random.rand() * Dist * np.sign(np.random.rand() - 0.5)
                    else:
                        WSO_Pos = gbest[j] + np.random.rand() * Dist * np.sign(np.random.rand() - 0.5)
                        WSO_Positions[i, j] = (WSO_Pos + WSO_Positions[i - 1, j]) / 2 * np.random.rand()

        # Update fitness and losses
        for i in range(num_sharks):
            if np.all(WSO_Positions[i, :] >= lb) and np.all(WSO_Positions[i, :] <= ub):
                fit_acc_loss = fobj(WSO_Positions[i, :])
                fitness[i], losses[i] = fit_acc_loss  # Accuracy and loss

                if fitness[i] > np.max(fitness):  # Maximize accuracy
                    wbest[i, :] = WSO_Positions[i, :].copy()

                if fitness[i] > fmax0:  # Maximize accuracy
                    fmax0 = fitness[i]
                    gbest = wbest[i, :].copy()

        # Store accuracy, loss, and convergence data for plotting
        acc_curve[ite] = np.max(fitness)
        loss_curve[ite] = np.min(losses)
        ccurve[ite] = fmax0

        if ite > 1:
            plt.plot([ite - 1, ite], [ccurve[ite - 1], ccurve[ite]], 'b')
            plt.title('Convergence characteristic curve')
            plt.xlabel('Iteration')
            plt.ylabel('Best score obtained so far')
            plt.draw()
            plt.pause(0.01)

    # Plot optimization results
    plot_optimization_results(acc_curve, loss_curve, ccurve)

    return fmax0, gbest, ccurve

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Set Hyperparameter Bounds and Run WSO
# lb = np.array([0.0001, 10, 10])
# ub = np.array([0.1, 200, 200])
# dim = 3
# num_sharks = 10
# max_iter = 5
# 
# fmax0, gbest, ccurve = WSO(num_sharks, max_iter, lb, ub, dim, objective_function)
# print("Best Hyperparameters:", gbest)
# print("Best Accuracy:", fmax0)

import numpy as np

# Get the true labels from the test generator
test_labels = test_gen.classes

# Make predictions on the test set
predictions = model.predict(test_gen)
predicted_classes = np.round(predictions).astype(int).flatten()  # Convert probabilities to binary predictions

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
fig, ax = plt.subplots(figsize=(5, 5))

# Generate the heatmap with Seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=list(classes.values()), yticklabels=list(classes.values()), ax=ax, linewidths=0.5, linecolor='gray')

# Set the axis labels and title
ax.set_xlabel('Predicted Label', fontsize=14)
ax.set_ylabel('True Label', fontsize=14)
ax.set_title('Confusion Matrix for WSO-SE-MSF-CNN model', fontsize=16)

# Set the colorbar properties
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)

# Set the tick labels' font size
ax.tick_params(axis='both', labelsize=12)

# Display the figure
plt.show()

# Generate classification report
from sklearn.metrics import classification_report

report_HH = classification_report(test_labels, predicted_classes)

print(report_HH);

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the test set
predicted_probs = model_load.predict(test_gen)  # Predicted probabilities

# Calculate the FPR and TPR
fpr, tpr, thresholds = roc_curve(test_labels, predicted_probs)

# Calculate AUC
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for WSO-SE-MSF-CNN model')
plt.legend(loc='lower right')
plt.grid()
plt.show()

from sklearn.metrics import precision_recall_curve, auc

# Get true labels from the test generator
true_labels = test_gen.classes

# Get predicted probabilities for the test set
predicted_probs = model_load.predict(test_gen)  # Predicted probabilities

# Flatten the predicted probabilities to 1D array
predicted_probs_flat = predicted_probs.flatten()

# Calculate precision and recall
precision, recall, _ = precision_recall_curve(true_labels, predicted_probs_flat)

# Calculate AUC for PR curve
pr_auc = auc(recall, precision)

# Plotting the Precision-Recall curve
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, color='blue', label='Precision-Recall curve (area = {:.2f})'.format(pr_auc))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for WSO-SE-MSF-CNN model')
plt.legend(loc='lower left')
plt.grid()
plt.show()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Function to plot feature maps one by one
def plot_feature_maps_individually(model, layer_name, image):
    # Create an intermediate model that outputs the feature maps of the given layer
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get the feature maps
    feature_maps = intermediate_model.predict(image)

    # Number of feature maps
    num_feature_maps = feature_maps.shape[-1]

    # Plot each feature map individually
    for i in range(num_feature_maps):
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.title(f'Feature Map {i+1}')
        plt.axis('off')
        plt.show()

# Input shape
input_shape = (256, 256, 3)


layer_name = 'conv2d_2'

# Plot the feature maps one by one
plot_feature_maps_individually(model, layer_name, img_array)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Function to plot feature maps one by one
def plot_feature_maps_individually(model, layer_name, image):
    # Create an intermediate model that outputs the feature maps of the given layer
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Get the feature maps
    feature_maps = intermediate_model.predict(image)

    # Number of feature maps
    num_feature_maps = feature_maps.shape[-1]

    # Plot each feature map individually
    for i in range(num_feature_maps):
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.title(f'Feature Map {i+1}')
        plt.axis('off')
        plt.show()

# Input shape
input_shape = (256, 256, 3)



layer_name = 'conv2d'

# Plot the feature maps one by one
plot_feature_maps_individually(model, layer_name, img_array)

