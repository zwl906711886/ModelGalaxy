import os
import pickle

import clip
import numpy as np
import torch
from sklearn.cluster import KMeans
from torchvision.datasets import CIFAR100

# from load_cifar100_10c import load_cifar100_superclass

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)


# Extract and merge features
def make_task_meta_features(task_features: np.ndarray) -> np.ndarray:
    task_features_normalized = task_features.cpu().numpy()
    nfeature = np.vsplit(task_features_normalized, 5)
    for i in range(5):
        w = make_task_meta_features2(nfeature[i])
        if i == 0:
            ww = w
        else:
            ww = np.append(ww, w)
    return ww


def make_task_meta_features2(task_features: np.ndarray) -> np.ndarray:
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(task_features)
    centroids = kmeans.cluster_centers_
    wk = centroids[0]
    return wk

k = 0
# Preprocess dataset, description and extract the meta_feature, k means the task number.
dataset0 = load_cifar100_superclass(is_train=True, superclass_type='predefined', target_superclass_idx=k, n_classes=10,
                                    reorganize=True)
tmp = -1
wi_list = []
wt_list = []

# Input task description
text_inputs = torch.cat([clip.tokenize(
    f"photos of {dataset0.classes[0]}, {dataset0.classes[1]}, {dataset0.classes[2]}, {dataset0.classes[3]}, {dataset0.classes[4]}.")]).to(
    device)

# Input task dataset images and labels
for i in range(2500):
    image, class_id = dataset0[i]
    image_input1 = preprocess(image).unsqueeze(0).to(device)
    wi_list.append(image_input1)
    image_text_input1 = clip.tokenize(f"photos of {dataset0.classes[class_id]}.")
    wt_list.append(image_text_input1)
image_t_input = torch.cat(wi_list)
image_text_input = torch.cat(wt_list).to(device)
image_input = torch.cat(wi_list)

# Feature extraction and fusion
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    itext_feature = model.encode_text(image_text_input)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
text_features2 = text_features.cpu().numpy()
text_features3 = text_features2[0]
task_meta_features = make_task_meta_features(image_features)
text_features4 = make_task_meta_features(itext_feature)

# Merge image features and description features
f_meta_features = 0.5 * task_meta_features + 0.5 * text_features4
f_meta_features = np.concatenate((f_meta_features, text_features3), axis=0)

# print(f_meta_features)
# Save the feature
with open(f'task_embbeding_path', 'wb') as f:
    pickle.dump(f_meta_features, f)
