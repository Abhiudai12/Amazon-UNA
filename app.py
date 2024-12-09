from flask import Flask, request, render_template
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokError
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

NGROK_KEY = os.getenv("NGROK_KEY")
API_KEY = os.getenv("API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

ngrok.set_auth_token(NGROK_KEY)

client = MongoClient(MONGO_URI)

db = client['product_analyzer']  
history_collection = db['question_answers']  

db.question_answers.update_many(
    {"timestamp": {"$exists": False}},  
    {"$set": {"timestamp": pd.Timestamp.now()}}  
)

app = Flask(__name__)
analyzer = ProductAnalyzer()

PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    asin = request.form['asin']
    question = request.form['question']
    api_key = API_KEY

    product_info = analyzer.get_product_info(asin, api_key)
    if not product_info:
        error_message = "Failed to retrieve product information. Check ASIN or API key."
        return render_template('index.html', error=error_message)

    sentiment_data = []
    for review in product_info['reviews']:
        sentiment = analyzer.analyze_sentiment(review)
        sentiment_data.append(sentiment)

    bert_answer = analyzer.get_bert_answer(question, product_info['analysis_text'])
    flan_answer = analyzer.get_flan_answer(question, product_info['analysis_text'])
    combined_answer = analyzer.combine_answers(bert_answer, flan_answer)

    try:
        history_collection.insert_one({
            "asin": asin,
            "question": question,
            "answer": combined_answer,
            "product_title": product_info['title'],
            "timestamp": pd.Timestamp.now()  
        })
        print("Data inserted successfully into MongoDB.")
    except Exception as e:
        print(f"Error inserting data into MongoDB: {e}")

    create_sentiment_plot(sentiment_data)
    create_word_heatmap(product_info['reviews'])

    sentiment_summary = {
        "positive": sum(1 for s in sentiment_data if s['sentiment'] == 'POSITIVE'),
        "negative": sum(1 for s in sentiment_data if s['sentiment'] == 'NEGATIVE'),
        "neutral": sum(1 for s in sentiment_data if s['sentiment'] == 'NEUTRAL'),
    }

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

    history = list(history_collection.find().sort("timestamp", -1))  
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
    stopwords = set(STOPWORDS)
    word_counts = Counter(
        word.strip('.,!?()[]{}').lower()
        for review in reviews
        for word in review.split()
        if word.lower() not in stopwords
    )
    common_words = word_counts.most_common(20)  
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
    try:

        public_url = ngrok.connect(5000)
        print(f"ngrok tunnel URL: {public_url}")
    except PyngrokNgrokError as e:
        print(f"Error with ngrok: {e}")
        print("Ensure no other ngrok sessions are active or check your ngrok configuration.")

    app.run(debug=True)