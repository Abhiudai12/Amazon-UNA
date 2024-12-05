import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from typing import List, Dict, Optional  # Add Optional import here
from transformers import pipeline, BertForQuestionAnswering, BertTokenizerFast, T5Tokenizer, T5ForConditionalGeneration
import requests
import re
import torch


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

class ProductAnalyzer:
    def __init__(self):
        # Initialize models
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.bert_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.bert_model = BertForQuestionAnswering.from_pretrained(self.bert_model_name)
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(self.bert_model_name)
        self.flan_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        self.flan_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_product_info(self, asin: str, api_key: str) -> Optional[Dict]:
        """Fetch high-quality product information and reviews from Amazon API."""
        product_details_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
        product_reviews_url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"

        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
        }

        try:
            # Fetch product details
            product_response = requests.get(
                product_details_url,
                headers=headers,
                params={"asin": asin, "country": "IN"}
            )

            # Check response status
            if product_response.status_code != 200:
                print(f"Product details request failed: {product_response.status_code}")
                print(f"Response: {product_response.text}")
                return None

            product_data = product_response.json().get('data', {})
            if not product_data:
                print("No product data returned from API")
                return None

            # Extract product information
            product_info = {
                'title': product_data.get('product_title', 'Title not available'),
                'description': product_data.get('product_description', 'Description not available'),
                'about': ' '.join(product_data.get('about_product', [])) or 'No details available',
                'details': product_data.get('product_details', 'No product details available'),
                'price': product_data.get('product_price', 'Price not available'),
                'rating': product_data.get('product_star_rating', 'Rating not available'),
                'reviews_count': product_data.get('product_num_ratings', 0),
                'dimensions': product_data.get('product_dimensions', 'Dimension data not available')
            }

            # Fetch product reviews
            reviews_response = requests.get(
                product_reviews_url,
                headers=headers,
                params={"asin": asin, "country": "IN"}
            )

            # Initialize reviews_texts to handle cases where no reviews are fetched
            reviews_texts = []
            if reviews_response.status_code == 200:
                reviews_data = reviews_response.json().get('data', {}).get('reviews', [])
                if reviews_data:
                    reviews_texts = [
                        review.get('review_comment', 'Review not available')
                        for review in reviews_data[:10]  # Fetch up to 10 reviews
                    ]
                else:
                    print("No reviews returned from API")
            else:
                print(f"Product reviews request failed: {reviews_response.status_code}")
                print(f"Response: {reviews_response.text}")

            # Combine text for sentiment analysis and question answering
            combined_texts = [
                f"Title: {product_info['title']}",
                f"Price: {product_info['price']}",
                f"Rating: {product_info['rating']} ({product_info['reviews_count']} reviews)",
                f"Details: {product_info['details']}",
                f"About: {product_info['about']}",
                f"Description: {product_info['description']}",
                f"Dimensions: {product_info['dimensions']}",
            ]

            # Add reviews to the combined text
            combined_texts += reviews_texts

            # Compress combined text to fit within 512 characters for analysis
            compressed_text = self.compress_to_limit(combined_texts, limit=512)
            product_info['analysis_text'] = self.clean_text(compressed_text)

            # Store reviews for sentiment analysis and visualization
            product_info['reviews'] = reviews_texts

            return product_info

        except requests.exceptions.RequestException as req_err:
            print(f"Network error fetching product data: {str(req_err)}")
            return None
        except ValueError as val_err:
            print(f"Data processing error: {str(val_err)}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching product data: {str(e)}")
            return None


    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text."""
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            return {
                'sentiment': result['label'],
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'sentiment': 'UNKNOWN', 'confidence': 0.0}

    def get_bert_answer(self, question: str, context: str) -> str:
        """Get answer using BERT model."""
        try:
            inputs = self.bert_tokenizer.encode_plus(
                question,
                context,
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                start_scores = outputs.start_logits
                end_scores = outputs.end_logits

            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)

            if end_index < start_index:
                end_index = start_index

            answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
            answer = self.bert_tokenizer.decode(answer_tokens, skip_special_tokens=True)

            return answer.strip() if answer.strip() else "No clear answer found."

        except Exception as e:
            print(f"Error in BERT answer generation: {str(e)}")
            return "Error generating answer"

    def get_flan_answer(self, question: str, context: str) -> str:
        """Get answer using FLAN-T5 model."""
        try:
            input_text = f"question: {question} context: {context}"
            inputs = self.flan_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = self.flan_model.generate(**inputs, max_length=100)

            return self.flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            print(f"Error in FLAN answer generation: {str(e)}")
            return "Error generating answer"

    def compress_to_limit(self, texts: List[str], limit: int = 512) -> str:
        """Compress and combine multiple texts within a character limit."""
        compressed_text = ""
        for text in texts:
            if len(compressed_text) + len(text) <= limit:
                compressed_text += f"{text} "
            else:
                # Truncate the last part if it exceeds the limit
                remaining_space = limit - len(compressed_text)
                compressed_text += text[:remaining_space]
                break
        return compressed_text.strip()

    def combine_answers(self, bert_answer: str, flan_answer: str) -> str:
        """Combine answers intelligently, avoiding repetition and irrelevance."""
        if bert_answer.lower() == flan_answer.lower():
            return bert_answer.strip()
        if bert_answer.lower() in flan_answer.lower():
            return flan_answer.strip()
        if flan_answer.lower() in bert_answer.lower():
            return bert_answer.strip()
        return f"{bert_answer.strip()} / {flan_answer.strip()}"

    def generate_context_aware_response(self, question: str, answer: str, sentiment_info: Dict) -> str:
        """Generate a response that considers sentiment context for e-commerce site."""

        question_lower = question.lower()
        sentiment = sentiment_info['sentiment']
        confidence = sentiment_info['confidence']

        # Generate a review summary based on sentiment
        review_summary = ""
        if sentiment == 'POSITIVE':
            review_summary = "The customer reviews are overwhelmingly positive, highlighting the product's strengths."
        elif sentiment == 'NEGATIVE':
            review_summary = "The customer reviews indicate some concerns, with a few users experiencing issues."
        else:
            review_summary = "The customer reviews are mixed, with both positive and neutral feedback."

        # Define response templates for different categories of questions
        response_templates = {
            'worth': f"{review_summary} {answer} Additionally, many customers believe this product is a great value for the price.",
            'good': f"{review_summary} {answer} Users generally praise this feature for its quality and functionality.",
            'problem': f"{review_summary} {answer} However, a few users reported challenges with this aspect of the product.",
            'recommend': f"{review_summary} {answer} Based on customer feedback, many recommend this product.",
            'default': f"{review_summary} {answer} Thank you."
        }

        # Categorize the question to choose the correct response template
        if any(word in question_lower for word in ['worth', 'buy', 'purchase', 'recommend']):
            category = 'worth'
        elif any(word in question_lower for word in ['good', 'great', 'best', 'quality']):
            category = 'good'
        elif any(word in question_lower for word in ['problem', 'issue', 'concern', 'worry']):
            category = 'problem'
        elif 'recommend' in question_lower:
            category = 'recommend'
        else:
            category = 'default'

        template = response_templates.get(category, response_templates['default'])
        return template.format(answer=answer)

    def analyze_product(self, asin: str, api_key: str):
        """Main product analysis pipeline."""
        # Fetch product information
        product_info = self.get_product_info(asin, api_key)
        if not product_info:
            print("Failed to fetch product information.")
            return

        # Print basic product details
        print(f"Product Title: {product_info['title']}")
        print(f"Product Details: {product_info['details']}")
        print(f"Price: {product_info['price']}")

        # Sentiment Analysis
        total_confidence = 0
        sentiment_count = 0

        # Iterate over reviews and perform sentiment analysis
        for review in product_info['reviews']:
            sentiment = self.analyze_sentiment(review)  # Perform sentiment analysis on each review

            total_confidence += sentiment['confidence']
            sentiment_count += 1

        print("The customer sentiment for this product is ",sentiment['sentiment']," with", end="")
        # Calculate the average sentiment confidence
        if sentiment_count > 0:
            average_confidence = total_confidence / sentiment_count
            print(f"Average Sentiment Confidence: {average_confidence:.2f}")
        else:
            print("No reviews for sentiment analysis.")


        while True:
            question = input("\nEnter your question about the product (or 'exit' to quit): ").strip()
            if question.lower() == 'exit':
                break

            # Get answers from BERT and FLAN models
            bert_answer = self.get_bert_answer(question, product_info['analysis_text'])
            flan_answer = self.get_flan_answer(question, product_info['analysis_text'])

            # Combine the answers
            combined_answer = self.combine_answers(bert_answer, flan_answer)

            # Print individual answers and the combined answer (for debugging/insights if needed)
            print(f"BERT Answer: {bert_answer}")
            print(f"FLAN-T5 Answer: {flan_answer}")


            # Generate context-aware response using the combined answer
            response = self.generate_context_aware_response(question, combined_answer, sentiment)
            print(f"\nFinal Response: {response}")

def main():
    analyzer = ProductAnalyzer()

    results = analyzer.analyze_product(asin, question, api_key)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print("\nProduct Analysis:")
    print("-" * 50)
    print(f"Product: {results['product_info']['title']}")
    print(f"Price: {results['product_info']['price']}")
    print(f"Rating: {results['product_info']['rating']} ({results['product_info']['reviews_count']} reviews)")

    print("\nSentiment-Aware Combined Response:")
    print(results['combined_answer']['context_aware'])

    print("\nRaw Combined Answer:")
    print(results['combined_answer']['raw'])

if __name__ == "__main__":
    main()