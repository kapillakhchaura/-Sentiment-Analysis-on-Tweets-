import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Step 1: Load CSV and select useful columns
df = pd.read_csv(r"D:\internship\Tweets.csv\Tweets.csv")
df = df[['text', 'airline_sentiment']].dropna()

# Step 2: Preprocessing setup
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|\#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Show result
print(df[['text', 'clean_text']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Step 3: Text to TF-IDF vectors
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()

# Step 4: Labels and train-test split
y = df['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ TF-IDF vectorization complete")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Your step 1 to 4 code yahan pe already likha hua hai...

# 👇 Step 5: Train the model
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 👇 Step 6: Evaluate the model
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ✅ WordCloud for Positive Tweets
from wordcloud import WordCloud
import matplotlib.pyplot as plt

positive_text = " ".join(df[df['airline_sentiment'] == 'positive']['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("🌟 WordCloud: Positive Sentiment Tweets")
plt.show()

# ✅ Sample Prediction on custom tweets
sample_texts = ["I love the airline's service!", "Very rude staff and delay!", "Okay flight. Nothing special."]
sample_cleaned = [clean_text(text) for text in sample_texts]
sample_vectors = tfidf.transform(sample_cleaned).toarray()
sample_predictions = model.predict(sample_vectors)

for i, text in enumerate(sample_texts):
    print(f"📝 Tweet: {text}\n🔍 Predicted Sentiment: {sample_predictions[i]}\n")
