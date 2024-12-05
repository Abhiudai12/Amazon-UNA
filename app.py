from flask import Flask, request, render_template
import os
from model import ProductAnalyzer
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS
from collections import Counter

from pymongo import MongoClient

load_dotenv()

API_KEY = os.getenv("API_KEY")

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Access the database and the collection
db = client['product_analyzer']  # The name of the database
history_collection = db['question_answers']  # The collection to store questions and answers

# Update existing records to add a timestamp if missing
db.question_answers.update_many(
    {"timestamp": {"$exists": False}},  # Find records without a timestamp
    {"$set": {"timestamp": pd.Timestamp.now()}}  # Add the current timestamp to those records
)

app = Flask(__name__)
analyzer = ProductAnalyzer()

# Directory to save plots
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

@app.route('/')
def home():
    """Render the home page with input form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the product based on ASIN and user-provided question."""
    asin = request.form['asin']
    question = request.form['question']
    api_key = API_KEY

    # Step 1: Fetch product information
    product_info = analyzer.get_product_info(asin, api_key)
    if not product_info:
        error_message = "Failed to retrieve product information. Check ASIN or API key."
        return render_template('index.html', error=error_message)

    # Step 2: Perform sentiment analysis on reviews
    sentiment_data = []
    for review in product_info['reviews']:
        sentiment = analyzer.analyze_sentiment(review)
        sentiment_data.append(sentiment)

    # Step 3: Answer user question
    bert_answer = analyzer.get_bert_answer(question, product_info['analysis_text'])
    flan_answer = analyzer.get_flan_answer(question, product_info['analysis_text'])
    combined_answer = analyzer.combine_answers(bert_answer, flan_answer)

    try:
        history_collection.insert_one({
            "asin": asin,
            "question": question,
            "answer": combined_answer,
            "product_title": product_info['title'],
            "timestamp": pd.Timestamp.now()  # Ensure the timestamp is added
        })
        print("Data inserted successfully into MongoDB.")
    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")


    # Step 4: Generate plots
    create_sentiment_plot(sentiment_data)
    create_word_heatmap(product_info['reviews'])

    # Step 5: Summarize sentiment results
    sentiment_summary = {
        "positive": sum(1 for s in sentiment_data if s['sentiment'] == 'POSITIVE'),
        "negative": sum(1 for s in sentiment_data if s['sentiment'] == 'NEGATIVE'),
        "neutral": sum(1 for s in sentiment_data if s['sentiment'] == 'NEUTRAL'),
    }

    # Render results to the webpage
    return render_template(
        'index.html',
        product_info=product_info,
        sentiment_summary=sentiment_summary,
        combined_answer=combined_answer,
        sentiment_plot=f"{PLOT_DIR}/sentiment_plot.png",
        heatmap=f"{PLOT_DIR}/heatmap.png"
    )

@app.route('/history')
def history():
    """Render a page showing all previous questions and answers."""
    # Fetch all questions and answers from MongoDB
    history = list(history_collection.find().sort("timestamp", -1))  # Sort by timestamp (most recent first)
    return render_template('history.html', history=history)


def create_sentiment_plot(sentiment_data):
    """Create a bar plot for sentiment confidence scores."""
    review_indices = [f"Review {i+1}" for i in range(len(sentiment_data))]
    confidence_scores = [s['confidence'] for s in sentiment_data]
    colors = ['green' if s['sentiment'] == 'POSITIVE' else 'red' for s in sentiment_data]

    plt.figure(figsize=(10, 6))
    plt.bar(review_indices, confidence_scores, color=colors, alpha=0.7)
    plt.xlabel('Reviews', fontsize=12)
    plt.ylabel('Sentiment Confidence Score', fontsize=12)
    plt.title('Sentiment Confidence for Reviews', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/sentiment_plot.png")
    plt.close()

def create_word_heatmap(reviews):
    """Create a heatmap for the most common words in reviews."""
    stopwords = set(STOPWORDS)
    word_counts = Counter(
        word.strip('.,!?()[]{}').lower()
        for review in reviews
        for word in review.split()
        if word.lower() not in stopwords
    )
    common_words = word_counts.most_common(20)  # Fetch the 20 most common words
    words, counts = zip(*common_words)
    heatmap_data = pd.DataFrame({'Word': words, 'Count': counts}).pivot_table(values='Count', index='Word')

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt='.0f', linewidths=0.5)
    plt.title("Word Heatmap from Reviews", fontsize=14)
    plt.xlabel("Frequency", fontsize=12)
    plt.ylabel("Words", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/heatmap.png")
    plt.close()

if __name__ == "__main__":
    app.run(debug=True)
