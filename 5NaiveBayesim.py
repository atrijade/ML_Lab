import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

data = [
    ['ham', "Hey, are we still on for dinner tonight?"],
    ['spam', "WINNER! You have won a free cruise. Call now!"],
    ['ham', "I'll call you back in 10 minutes."],
    ['spam', "URGENT! Your account has been suspended. Click to verify."],
    ['ham', "Don't forget about the meeting at 3 PM."],
    ['spam', "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May"],
    ['ham', "Lunch tomorrow?"],
    ['spam', "Claim your free ringtone now by texting WIN to 80085"],
    ['ham', "Can you send me the report before noon?"],
    ['spam', "Congratulations! You've been selected for a $1000 Walmart gift card."]
]

df = pd.DataFrame(data, columns=["label", "text"])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42
)

vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("=== Evaluation ===")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")

test_messages = [
    "Your loan is approved! Call now to claim.",
    "Hi, just checking in to see how you're doing.",
    "Don't miss out on this limited-time offer!",
    "Are you joining the office call at 2 PM?"
]

test_vec = vectorizer.transform(test_messages)
predictions = model.predict(test_vec)

print("\n=== Predictions ===")
for msg, pred in zip(test_messages, predictions):
    print(f"'{msg}' => {'spam' if pred==1 else 'ham'}")