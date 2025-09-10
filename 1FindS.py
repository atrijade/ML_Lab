import csv

# Load CSV file
def loadCsv(filename):
    with open(filename, "r") as file:
        lines = csv.reader(file)
        dataset = list(lines)
    return dataset

# Attributes (features)
attributes = ['Sky', 'Temp', 'Humidity', 'Wind', 'Water', 'Forecast']
print("Attributes:", attributes)

# Load dataset from file
filename = "Weather.csv"  # Ensure this CSV file exists in the same directory
dataset = loadCsv(filename)
print("\nDataset:")
for row in dataset:
    print(row)

# Target values (assumed labels for each example)
target = ['Yes', 'Yes', 'No', 'Yes']
print("\nTarget:", target)

# Initialize hypothesis with most specific hypothesis
num_attributes = len(attributes)
hypothesis = ['0'] * num_attributes
print("\nInitial Hypothesis:", hypothesis)

# Apply Find-S algorithm
print("\nFinding hypothesis...\n")
for i in range(len(target)):
    if target[i] == 'Yes':
        for j in range(num_attributes):
            if hypothesis[j] == '0':
                hypothesis[j] = dataset[i][j]
            elif hypothesis[j] != dataset[i][j]:
                hypothesis[j] = '?'
        print(f"Step {i+1}: {hypothesis}")

# Final result
print("\nFinal Hypothesis:", hypothesis)
