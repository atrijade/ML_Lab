import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, export_text

data = pd.read_csv("decision_tree_data.csv")
print("Dataset:")
print(data)
print("\nDecision Tree:")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le = preprocessing.LabelEncoder()
X_enc = X.apply(lambda x: preprocessing.LabelEncoder().fit_transform(x))
y_enc = le.fit_transform(y)

model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1)
model.fit(X_enc, y_enc)

tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
