Sentiment Analysis on Tweets (CODTECH Internship - Task 4)

This project performs Sentiment Analysis on Twitter data using Natural Language Processing (NLP) techniques.  
It is developed as part of the CODTECH Data Analytics Internship (Task 4).


Dataset Used

Dataset: Tweets.csv
Source: [Kaggle Airline Twitter Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
Contains tweets labeled as positive, negative, or neutral



Technologies & Libraries Used

Python
Pandas
NumPy
NLTK (Natural Language Toolkit)
Scikit-learn (sklearn)
WordCloud
Matplotlib



Project Workflow

1. Data Loading – Loaded Tweets.csv containing tweet text and sentiment labels
2. Text Cleaning – 
   Removed URLs, mentions (@), hashtags, punctuation
   Converted text to lowercase
   Removed stopwords and lemmatized words
3. TF-IDF Vectorization – Converted clean text into numerical feature vectors
4. Model Training – Used Multinomial Naive Bayes for sentiment classification
5. Model Evaluation – Calculated accuracy, classification report, and confusion matrix
6. Visualization – Generated a WordCloud for positive sentiment tweets



Results

- Model Accuracy: ~73.9%
- Confusion Matrix & Classification Report included
- WordCloud shows top keywords from positive tweets


Sample Output

![WordCloud - Positive Tweets](path/to/your/image.png) <!-- replace with actual GitHub image path -->



Sample Predictions

python
Tweet: "I love the airline's service!"
Predicted Sentiment: positive

Tweet: "Very rude staff and delay!"
Predicted Sentiment: negative
