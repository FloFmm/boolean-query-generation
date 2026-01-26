from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.tree import export_text
import numpy as np
import pickle
import time

file = "/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data/bag_of_words/vectors,d=433660,mindf=10,maxdf=0.5,mesh=True.pkl"
features_file = "/data/horse/ws/flml293c-master-thesis/systematic-review-datasets/data/bag_of_words/feature_names,d=433660,mindf=10,maxdf=0.5,mesh=True.pkl"
with open(file, "rb") as f:
    X = pickle.load(f)
with open(features_file, "rb") as f:
    feature_names = pickle.load(f)

y_train = np.zeros(X.shape[0], dtype=int)
y_train[:50] = 1

text_clf = Pipeline(
    [
        ("clf", RandomForestClassifier(n_estimators=10, max_depth=5)),
    ]
)

st = time.time()
text_clf.fit(X, y_train)
print(time.time() - st)

predicted = text_clf.predict(X)

print(metrics.classification_report(y_train, predicted))
clf = text_clf.named_steps["clf"]
for i, tree in enumerate(clf.estimators_[:3]):  # print first 3 trees
    print(f"Tree {i}")
    print(export_text(tree, feature_names=feature_names))
