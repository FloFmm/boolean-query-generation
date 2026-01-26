# from sklearn.datasets import fetch_california_housing
from sklearn.metrics import precision_recall_curve

# from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# from imodels import SkopeRulesClassifier

# # Use modern dataset (load_boston was removed)
# dataset = fetch_california_housing()

# X, y = dataset.data, dataset.target > 3  # threshold to make it binary

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# clf = SkopeRulesClassifier(
#     n_estimators=30,
#     precision_min=0.2,
#     recall_min=0.01,
# )

# clf.fit(X_train, y_train, feature_names=dataset.feature_names)

# # Get rule-based risk score for test examples
# y_score = clf._score_top_rules(X_test)

# precision, recall, _ = precision_recall_curve(y_test, y_score)
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()


from sklearn.model_selection import train_test_split
from imodels import (
    get_clean_dataset,
    SkopeRulesClassifier,
)  # import any imodels model here

# prepare data (a sample clinical dataset)
X, y, feature_names = get_clean_dataset("csi_pecarn_pred")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# fit the model
model = SkopeRulesClassifier(
    n_estimators=30,
    precision_min=0.2,
    recall_min=0.01,
)  # initialize a tree model and specify only 4 leaf nodes
model.fit(X_train, y_train, feature_names=feature_names)  # fit model
preds = model.predict(X_test)  # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(
    X_test
)  # predicted probabilities: shape is (n_test, n_classes)
print(model.rules_[0])  # print the model


# y_score = model._predict_top_rules(X_test, 5)
# precision, recall, _ = precision_recall_curve(y_test, y_score)
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.show()
