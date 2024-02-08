import os
import pickle
import timm
import clip
import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision.datasets import CIFAR100
from sklearn.preprocessing import normalize
from load_dataset import load_cifar100_10c
from load_cifar100_10c import load_cifar100_superclass
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve, \
    validation_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model2, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

def make_features(task_features: np.ndarray) -> np.ndarray:
    task_features_normalized = task_features.cpu().numpy()
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(task_features_normalized)
    centroids = kmeans.cluster_centers_
    wk = centroids.mean(axis=0)
    return wk

# Preprocess image and extract the metafeature, k means the dataset number.
dataset0=load_cifar100_superclass(is_train=True, superclass_type='predefined', target_superclass_idx=k, n_classes=10, reorganize=True)
tmp = -1
wi_list = []
for i in range(2500):
    image, class_id = dataset0[i]
    image_input1 = preprocess(image).unsqueeze(0).to(device)
    wi_list.append(image_input1)
image_input = torch.cat(wi_list)
with torch.no_grad():
    image_features = model2.encode_image(image_input)
image_features /= image_features.norm(dim=-1, keepdim=True)
meta_features = make_features(image_features)
print("meta_feature.shape: ", meta_features.shape)

# Save the metafeature
with open(meta_feature_path, 'wb') as f:
    pickle.dump(meta_features, f)
