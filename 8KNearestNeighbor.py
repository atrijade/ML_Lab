import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame({
    'Height': [170, 165, 180, 175, 160, 155, 185, 170],
    'Weight': [70, 55, 80, 75, 50, 45, 90, 65],
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female']
})

print("Training Data:")
print(data)

X = data[['Height', 'Weight']]
y = data['Gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("\nTest Data:")
print(X_test)
print("\nActual Labels:")
print(y_test.values)
print("\nPredicted Labels:")
print(y_pred)

correct = X_test[y_test == y_pred]
wrong = X_test[y_test != y_pred]

print("\nCorrect Predictions:")
print(correct)
print("\nWrong Predictions:")
print(wrong)