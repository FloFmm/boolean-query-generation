from sklearn.datasets import load_boston
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
from skrules import SkopeRulesClassifier

dataset = load_boston()
clf = SkopeRulesClassifier(max_depth_duplication=None,
                 n_estimators=30,
                 precision_min=0.2,
                 recall_min=0.01,
                 feature_names=dataset.feature_names)

X, y = dataset.data, dataset.target > 25
X_train, y_train = X[:len(y)//2], y[:len(y)//2]
X_test, y_test = X[len(y)//2:], y[len(y)//2:]
clf.fit(X_train, y_train)
y_score = clf._score_top_rules(X_test) # Get a risk score for each test example
precision, recall, _ = precision_recall_curve(y_test, y_score)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')
plt.show()