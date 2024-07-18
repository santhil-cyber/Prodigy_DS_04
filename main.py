# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
url = 'https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis'
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Data Preprocessing
# Convert sentiment to categorical type
data['sentiment'] = data['sentiment'].astype('category')

# Visualize the distribution of sentiments
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Generate word clouds for each sentiment
sentiments = data['sentiment'].unique()
for sentiment in sentiments:
    text = ' '.join(data[data['sentiment'] == sentiment]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')
    plt.show()

# Sentiment analysis using a simple model (e.g., TextBlob)
from textblob import TextBlob

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

data['predicted_sentiment'] = data['text'].apply(get_sentiment)

# Compare actual vs predicted sentiments
plt.figure(figsize=(10, 6))
sns.countplot(x='predicted_sentiment', data=data, palette='viridis')
plt.title('Predicted Sentiment Distribution')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Count')
plt.show()
