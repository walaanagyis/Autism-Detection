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

# Ignor the error
# The first thing we should ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# wget https://www.kaggle.com/api/v1/datasets/download/sadikibrahim17/ytuia-2d -O ./ytuia-2d.zip

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# unzip ./ytuia-2d.zip -d ./ytuia-2d

import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models.resnet as resnet
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import glob
import shutil

# 1. Load pre-trained ResNet-50 model with weights
try:
    weights = resnet.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
except Exception as e:
    print(f"Error loading ResNet-50 weights: {e}")
    print("Falling back to model without weights...")
    model = models.resnet50(weights=None)

model.eval()  # Set to evaluation mode
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the final fully connected layer

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Extract features from images
def extract_features(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(img).squeeze().numpy()  # Extract features
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Data directories
train_dir = '/content/ytuia-2d/YTUIA_2D/Train'
test_dir = '/content/ytuia-2d/YTUIA_2D/Test'
train_image_paths = glob.glob(os.path.join(train_dir, '*.jpg'))
test_image_paths = glob.glob(os.path.join(test_dir, '*.jpg'))

# 3. Check for leakage in the current split
def check_leakage(train_paths, test_paths, sample_size=100):
    train_features = []
    valid_train_paths = []
    for img_path in train_paths[:sample_size]:  # Limit to sample_size for efficiency
        feature = extract_features(img_path)
        if feature is not None:
            train_features.append(feature)
            valid_train_paths.append(img_path)

    test_features = []
    valid_test_paths = []
    for img_path in test_paths[:sample_size]:  # Limit to sample_size for efficiency
        feature = extract_features(img_path)
        if feature is not None:
            test_features.append(feature)
            valid_test_paths.append(img_path)

    if len(train_features) == 0 or len(test_features) == 0:
        print("Not enough valid features to check leakage.")
        return None, None, None

    train_features = np.array(train_features)
    test_features = np.array(test_features)
    similarity_matrix = cosine_similarity(train_features, test_features)
    max_similarity = similarity_matrix.max()
    print(f"Maximum similarity between train and test: {max_similarity}")
    print(f"Is there leakage? {'Yes' if max_similarity >= 0.95 else 'No'}")
    return max_similarity, valid_train_paths, valid_test_paths

# Check leakage in the current split
print("Checking leakage in the current train/test split...")
max_similarity, valid_train_paths, valid_test_paths = check_leakage(train_image_paths, test_image_paths)

# 4. Re-cluster and re-split if leakage is detected
if max_similarity is None or max_similarity >= 0.95:
    print("Leakage detected or unable to check. Re-clustering and re-splitting data...")

    # Combine all images for clustering
    all_image_paths = train_image_paths + test_image_paths
    all_features = []
    valid_image_paths = []

    for img_path in all_image_paths:
        feature = extract_features(img_path)
        if feature is not None:
            all_features.append(feature)
            valid_image_paths.append(img_path)

    all_features = np.array(all_features)

    # Apply K-Means clustering
    num_clusters = 150  # Keep number of clusters
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(all_features)

    # Group images by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(valid_image_paths[idx])

    print(f"Number of K-Means clusters formed: {len(clusters)}")

    # Split clusters
    train_ratio, val_ratio, test_ratio = 0.75, 0.15, 0.10  # Adjusted for more test images
    total_clusters = len(clusters)
    train_count = int(train_ratio * total_clusters)
    val_count = int(val_ratio * total_clusters)

    cluster_ids = list(clusters.keys())
    np.random.shuffle(cluster_ids)

    train_clusters = cluster_ids[:train_count]
    val_clusters = cluster_ids[train_count:train_count + val_count]
    test_clusters = cluster_ids[train_count + val_count:]

    # Collect images for each split
    train_images = [img for cid in train_clusters for img in clusters[cid]]
    val_images = [img for cid in val_clusters for img in clusters[cid]]
    test_images = [img for cid in test_clusters for img in clusters[cid]]

    print(f"New split - Number of training images: {len(train_images)}")
    print(f"New split - Number of validation images: {len(val_images)}")
    print(f"New split - Number of test images: {len(test_images)}")

    # Verify leakage in the new split
    print("Verifying leakage in the new split...")
    train_features = []
    valid_train_paths = []
    for img_path in train_images[:100]:  # Limit to sample_size for efficiency
        feature = extract_features(img_path)
        if feature is not None:
            train_features.append(feature)
            valid_train_paths.append(img_path)

    test_features = []
    valid_test_paths = []
    for img_path in test_images[:100]:  # Limit to sample_size for efficiency
        feature = extract_features(img_path)
        if feature is not None:
            test_features.append(feature)
            valid_test_paths.append(img_path)

    if len(train_features) > 0 and len(test_features) > 0:
        train_features = np.array(train_features)
        test_features = np.array(test_features)
        similarity_matrix = cosine_similarity(train_features, test_features)
        new_max_similarity = similarity_matrix.max()
        print(f"Maximum similarity in new split: {new_max_similarity}")
        print(f"Is there leakage in new split? {'Yes' if new_max_similarity >= 0.95 else 'No'}")

        # Manual reassignment of highly similar images
        if new_max_similarity >= 0.90:  # Keep strict threshold
            print("High similarity detected. Reassigning highly similar images...")
            similarity_matrix = cosine_similarity(train_features, test_features)
            high_similarity_pairs = np.where(similarity_matrix >= 0.90)

            # Move test images with high similarity to training
            reassigned_images = []
            for i, j in zip(high_similarity_pairs[0], high_similarity_pairs[1]):
                if j < len(valid_test_paths):  # Ensure index is valid
                    test_img = valid_test_paths[j]
                    if test_img in test_images and test_img not in reassigned_images:
                        train_images.append(test_img)
                        test_images.remove(test_img)
                        reassigned_images.append(test_img)

            print(f"Reassigned {len(reassigned_images)} images from test to train.")
            print(f"Updated - Number of training images: {len(train_images)}")
            print(f"Updated - Number of validation images: {len(val_images)}")
            print(f"Updated - Number of test images: {len(test_images)}")

            # Re-verify leakage after reassignment
            print("Re-verifying leakage after reassignment...")
            train_features = [f for f in [extract_features(img) for img in train_images[:100]] if f is not None]
            test_features = [f for f in [extract_features(img) for img in test_images[:100]] if f is not None]

            if len(train_features) > 0 and len(test_features) > 0:
                train_features = np.array(train_features)
                test_features = np.array(test_features)
                similarity_matrix = cosine_similarity(train_features, test_features)
                final_max_similarity = similarity_matrix.max()
                print(f"Final maximum similarity in new split: {final_max_similarity}")
                print(f"Is there leakage in final split? {'Yes' if final_max_similarity >= 0.95 else 'No'}")
            else:
                print("Not enough valid features to re-verify leakage.")
    else:
        print("Not enough valid features to verify leakage in new split.")

    # Save new splits
    # /content/drive/MyDrive/ytuia-2d/YTUIA_2D
    output_dir = '/content/drive/MyDrive/ytuia-2d/YTUIA_2D'
    def save_split(images, split_name, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for img_path in images:
            shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))

    save_split(train_images, 'train', os.path.join(output_dir, 'train'))
    save_split(val_images, 'val', os.path.join(output_dir, 'val'))
    save_split(test_images, 'test', os.path.join(output_dir, 'test'))
else:
    print("No significant leakage detected in the current split.")
