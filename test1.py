import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load a small dataset from 20 Newsgroups
data = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey', 'sci.space', 'comp.graphics'], remove=('headers', 'footers', 'quotes'))

# Show sample data (first 5 rows)
df = pd.DataFrame({'Text': data.data, 'Category': [data.target_names[i] for i in data.target]})
print("Sample Data:")
print(df)
for i in range(3):
    print(f"Article {i+1} (Category: {data.target_names[data.target[i]]}):")
    print(data.data[i][:300] + "...\n")  # Print only first 300 characters
    

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(data.data)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Accuracy
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM")
plt.show()

# Word Frequency Plot
word_counts = pd.DataFrame({'Word': vectorizer.get_feature_names_out(), 'Importance': X_train.sum(axis=0).A1})
word_counts = word_counts.sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Word', data=word_counts)
plt.title("Top 10 Words by Importance")
plt.show()