> **Note**: All ML and data science work in this project is done by me. The original repo was created by my teammate, who did frontend part.
> This fork contains the complete implementation with my updates.


> 🏆 **Recognition:** This project received an **AA Grade** from professors at IIIT Nagpur  .
## About the project
" Analyzed REALTIME  Amazon Customer Reviews data, made a sentiment based Q & A model with similarity score of 73% ."

Amazon UNA - Product Analysis Web App\
Generates Sentiment based response with high accuracy\
Based on combined BERT and FLAN Answering Model\
Incorporated Sentiment analysis as our uniqueness\
Techologies Used : Rapid API(for fethcing product information from #Amazon) and ngrok(for deployment)

--
##Model Output :
<pre>  Enter your question about the product (or 'exit' to quit): Is it worth buying?
  BERT Answer: price : 843  FLAN-T5 Answer: Yes 
  Final Response: The customer reviews are overwhelmingly positive, highlighting the product's strengths. price : 843 / Yes Additionally, many customers believe this product is a great value for the price.  </pre>
### Result Screenshots :


Amazon Reviews fetched-


![Amazon Reviews fetched](images/Extracted_reviews.jpg)


Simmilarity Comparision-


![Screenshot 1](images/una1.png) 


Positve & Negative Sentiment Captured-


![Screenshot 2](images/una2.png)

## Steps to Run

Step 1 - Create a .env file with the following keys

`API_KEY` - Rapid API Key for Real Time Amazon Data\
`MONGO_URI` - Mongo Server Connection String\
`NGROK_KEY` - ngrok authentication token

Step 2 - Run this command

```
pip install -r requirements.txt
python app.py
```

