from imodels import FIGSClassifier, get_clean_dataset
from sklearn.model_selection import train_test_split

# prepare data (in this a sample clinical dataset)
X, y, feat_names = get_clean_dataset('csi_pecarn_pred')
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.33, random_state=42)

# fit the model
model = FIGSClassifier(max_rules=4)  # initialize a model
model.fit(X_train, y_train)   # fit model
preds = model.predict(X_test) # discrete predictions: shape is (n_test, 1)
preds_proba = model.predict_proba(X_test) # predicted probabilities: shape is (n_test, n_classes)

# visualize the model
model.plot(feature_names=feat_names, filename='out.svg', dpi=300)