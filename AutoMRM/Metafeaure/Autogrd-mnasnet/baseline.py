import os
import pickle
import timm
import clip
import torch
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from torchvision.datasets import CIFAR100
from sklearn.preprocessing import normalize
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
m = timm.create_model('mnasnet_100', pretrained=True).to(device)
m.reset_classifier(0)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

#Preprocess the image with pre-train model and random forest classification
output_path = 'output12/dataset.txt'
wy_list = []
dataset0 = load_cifar100_superclass(is_train=True, superclass_type='predefined', target_superclass_idx=k, n_classes=10,
                                    reorganize=True)
tmp = -1
wi_list = []
for i in range(2500):
    image, class_id = dataset0[i]
    image_input1 = preprocess(image).unsqueeze(0).to(device)
    wy_list.append(class_id)
    wi_list.append(image_input1)
image_input = torch.cat(wi_list)
with torch.no_grad():
    image_features = m(image_input)
image_features /= image_features.norm(dim=-1, keepdim=True)
task_meta_features = image_features
f_meta_features = task_meta_features
listData = f_meta_features.cpu().numpy().tolist()
wd_list.append(f_meta_features)
df = pd.DataFrame(listData)
dfy = pd.DataFrame(wy_list)
X = df
Y = dfy[0]
clf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=50)
clf.fit(X, Y)
result = clf.apply(X)
result = result.transpose()
result = pd.DataFrame(result)
result.to_csv(output_path, index=False, header=False)
