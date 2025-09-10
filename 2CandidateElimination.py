import pandas as pd

def candidate_elimination(data, domains):
    # Step 1: Initialize S and G
    S = ["0"] * (len(data[0]) - 1)
    G = [["?"] * (len(data[0]) - 1)]

    print("Initial Specific hypothesis S:", S)
    print("Initial General hypothesis G:", G)

    # Step 2: Loop through all training examples
    for row in data:
        example = row[:-1]           # all attributes
        label = row[-1].lower()      # target label

        if label == "yes":
            # Update Specific hypothesis S
            for j in range(len(S)):
                if S[j] == "0":
                    S[j] = example[j]
                elif S[j] != example[j]:
                    S[j] = "?"

            # Remove inconsistent hypotheses from G
            new_G = []
            for g in G:  # check each hypothesis in G
                ok = True
                for j in range(len(g)):  # check each attribute
                    if not (g[j] == "?" or g[j] == example[j]):
                        ok = False  # this hypothesis doesn't cover the example
                        break
                if ok:
                    new_G.append(g)  # keep it
            G = new_G

        else:  # Negative example
            new_G = []
            for g in G:
                for j in range(len(g)):
                    if g[j] == "?":
                        # Specialize using all possible values except the negative one
                        for val in domains[j]:
                            if val != example[j]:
                                new_hypothesis = g.copy()
                                new_hypothesis[j] = val
                                if new_hypothesis not in new_G:
                                    new_G.append(new_hypothesis)
            G = new_G

        # Print updates after each example
        print("\nTraining example:", row)
        print("Updated Specific hypothesis S:", S)
        print("Updated General hypothesis G:", G)

    return S, G


# Load CSV file
df = pd.read_csv("train.csv")
data = df.values.tolist()

# Extract domains of each attribute (all unique values per column, except label)
domains = [list(df[col].unique()) for col in df.columns[:-1]]

# Run Candidate Elimination
S_final, G_final = candidate_elimination(data, domains)

print("\nFinal Specific hypothesis:", S_final)
print("Final General hypothesis:", G_final)