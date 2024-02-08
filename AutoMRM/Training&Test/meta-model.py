import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import metric2

# The path saves real accuracy of all models on different tasks
path = 'result.csv'
df = pd.DataFrame(pd.read_csv(path, header=None))


# The model represent one DNN model, the SDS_acc means the probability of algorithm prediction,
# the real_acc represents the true accuracy of the model on the dataset, used to evaluate algorithms
class Model(object):
    def __init__(self, name, SDS_acc, real_acc):
        self.name = name
        self.SDS_acc = SDS_acc
        self.real_acc = real_acc


# Define the meta features (wd_list) and meta targets (y_list), y_list is extracted from evaluation results manually
# according to different tasks and fill here.
wd_list = []
y_list = []
leave_one_out = LeaveOneOut()

# Read meta features from previous meta feature files, the file_path is the path of the previous meta feature files.
# Read the meta features of each task separately and integrate them.
meta_features = []
for k in range(160):
    tmp_feature = []
    file_path = "test_{}/task_embbeding.pkl".format(k)
    with open(file_path, "rb") as f:
        tmp_feature.append(pickle.load(f))
    tmp_arr = np.array(tmp_feature[0]).flatten()
    meta_features.append(tmp_arr)
wd_list = meta_features

#  Format meta features and meta targets.
wd_list = np.array(wd_list)
y_list = np.array(y_list)

#  Split the training and testing sets into 4:1 and repeat 100 times to obtain experimental results
acc_sum = 0.0
acc_list = []
ndcg5_t = []
mrr_t = []
map_t = []
# Select the meta learner to use Ensemble learning or not according to the task
model = BaggingClassifier(base_estimator=SVC(probability=True),
                          max_samples=0.8,
                          max_features=0.8,
                          n_estimators=100,
                          bootstrap_features=True, n_jobs=-1)
# model = SVC(kernel='poly',probability=True,degree= 5)
pipe_lr = make_pipeline(StandardScaler(), model)
for i in range(100):
    indices = np.arange(y_list.shape[0])
    [indices_train, indices_test, y_train, y_test] = \
        train_test_split(indices, y_list, test_size=0.20, stratify=y_list)
    X_train_list = []
    X_test_list = []
    for j in indices_train:
        X_train_list.append(wd_list[j])
    X_train = np.array(X_train_list)
    for k in indices_test:
        X_test_list.append(wd_list[k])
    X_test = np.array(X_test_list)
    pipe_lr.fit(X_train, y_train)
    prediction_pro = pipe_lr.predict_proba(X_test)

    res = df.iloc[indices_test]
    res = res.reset_index(drop=True)
    ndcg5 = []
    mrr = []
    map = []
    for x in range(len(prediction_pro)):
        model_list = []
        for y in range(len(prediction_pro[0])):
            model_name = y
            mSDS_acc = prediction_pro[x][y]
            mReal_acc = res[y][x]
            model = Model(name=model_name, SDS_acc=mSDS_acc, real_acc=mReal_acc)
            model_list.append(model)
        print(model_list)
        ndcg5.append(metric2.NDCG(model_list, 5, 1)[0])
        mrr.append(metric2.MRR(model_list, 1)[0])
        map.append(metric2.MAP(model_list, 3, 1)[0])
    tmp = pipe_lr.score(X_test, y_test)
    acc_sum += tmp
    acc_list.append(tmp)
    ndcg5_t.append(sum(ndcg5) / len(ndcg5))
    mrr_t.append(sum(mrr) / len(mrr))
    map_t.append(sum(map) / len(map))
acc_avg = acc_sum / len(acc_list)
print("acc: ", acc_avg)
print("ndcg5: ", sum(ndcg5_t) / len(ndcg5_t))
print("mrr: ", sum(mrr_t) / len(mrr_t))
print("map: ", sum(map_t) / len(map_t))

# Save experimental results
with open('acc_list_10t.pkl', 'wb') as e1:
    pickle.dump(acc_list, e1)

with open('ndcg5_list_10t.pkl', 'wb') as e2:
    pickle.dump(ndcg5_t, e2)

with open('mrr_list_10t.pkl', 'wb') as e2:
    pickle.dump(mrr_t, e2)

with open('map_list_10t.pkl', 'wb') as e2:
    pickle.dump(map_t, e2)
