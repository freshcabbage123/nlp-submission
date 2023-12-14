import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Data
data = pd.read_csv('./stock_data.csv')

# Preprocessing
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['processed_text'] = data['Text'].apply(preprocess_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['processed_text']).toarray()
y = data['Sentiment']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Displaying Original vs Processed Text
comparison_table = data[['Text', 'processed_text']].head()
print(comparison_table)
