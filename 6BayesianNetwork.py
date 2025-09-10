import numpy as np
from sklearn.naive_bayes import MultinomialNB

age = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
gender = {'Male': 0, 'Female': 1}
family_history = {'Yes': 0, 'No': 1}
diet = {'High': 0, 'Medium': 1, 'Low': 2}
lifestyle = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedentary': 3}
cholesterol = {'High': 0, 'BorderLine': 1, 'Normal': 2}

X = np.array([
    [0,1,0,1,3,0], [1,0,0,0,2,0], [2,0,0,0,3,0],
    [1,1,0,1,2,0], [2,1,0,0,2,0], [3,0,1,2,1,1],
    [4,1,1,2,2,2], [4,1,1,2,3,2], [3,1,1,2,3,1], [2,0,1,2,2,1]
])
y = np.array([0,0,0,0,0,1,1,1,1,1])

model = MultinomialNB()
model.fit(X, y)

print("Enter patient medical data as numbers corresponding to the categories:")

try:
    a = int(input(f"Age {age}: "))
    g = int(input(f"Gender {gender}: "))
    f = int(input(f"Family History {family_history}: "))
    d = int(input(f"Diet {diet}: "))
    l = int(input(f"Lifestyle {lifestyle}: "))
    c = int(input(f"Cholesterol {cholesterol}: "))

    patient = np.array([[a, g, f, d, l, c]])
    prob = model.predict_proba(patient)[0]
    prediction = "Yes" if prob[0] > prob[1] else "No"

    print("\n=== Heart Disease Diagnosis ===")
    print(f"Probability of Heart Disease: {prob[0]:.2f}")
    print(f"Probability of No Heart Disease: {prob[1]:.2f}")
    print(f"Final Prediction: {prediction}")

except Exception as e:
    print("Invalid input. Please enter valid numbers.", e)