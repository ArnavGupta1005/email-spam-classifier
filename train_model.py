#Importing necessary libraries
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load and Prepare Data
print("Loading data from spam.csv...")
df = pd.read_csv('spam.csv', encoding='latin-1')
print("Data loaded successfully.")

#Split Data
print("Splitting data into training and testing sets...")
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split complete.")

#Create and Train the Model Pipeline
print("Training the model pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', MultinomialNB())
])

pipeline.fit(X_train, y_train)
print("Model training complete.")

#Evaluate the Model
print("Evaluating the model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
print(f"\nModel Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
print("\nIMPORTANT: Update the metrics in app.py with these new values!")

#Save All Necessary Assets
print("Saving the trained model pipeline...")
joblib.dump(pipeline, 'spam_classifier_pipeline.pkl')
print("Pipeline saved as spam_classifier_pipeline.pkl")

# Generate and save the Confusion Matrix plot
print("Generating and saving confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

# Generate and save Word Clouds
print("Generating and saving word clouds...")
spam_text = " ".join(df[df['Category'] == 'spam']['Message'])
ham_text = " ".join(df[df['Category'] == 'ham']['Message'])
spam_wordcloud = WordCloud(width=800, height=400, background_color='black').generate(spam_text)
ham_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
spam_wordcloud.to_file('spam_wordcloud.png')
ham_wordcloud.to_file('ham_wordcloud.png')
print("Word clouds saved successfully.")

print("\n--- All assets have been created successfully! ---")