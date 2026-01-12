import sys
import inspect
sys.path.insert(0, "/home/florian/Data/dev/scikit-learn")
from sklearn.tree import DecisionTreeClassifier as MyDecTree
from sklearn.tree import export_text
from scipy.sparse import csr_matrix
print("source file:", inspect.getsourcefile(MyDecTree))
# Simple dataset: x3 and (x1 or x2)
X = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 0]
]
y = [0, 1, 0, 1, 0, 0, 1, 0]  # labels

# Create the tree
tree = MyDecTree(
    splitter="best_or",
    max_depth=3,
    random_state=42,
    class_weight=1.0
)

# Fit the model
X_sparse = csr_matrix(X)
tree.fit(X_sparse, y)

# Make predictions
preds = tree.predict(X)

print("Predictions:", preds)
print("Tree depth:", tree.get_depth())
print("Number of leaves:", tree.get_n_leaves())
tree_text = export_text(tree, feature_names=["x1", "x2", "x3"])
print(tree_text)
