> **Note**: The original repo was created by my teammate, who did frontend part.
> This fork contains the complete implementation with my updates.
> Got 'AA' graade from IIIT professors for this project
## About the project
" Analyzed REALTIME  Amazon Customer Reviews data, made a sentiment based Q & A model with similarity score of 73% ."

Amazon UNA - Product Analysis Web App\
Generates Sentiment based response with high accuracy\
Based on combined BERT and FLAN Answering Model\
Incorporated Sentiment analysis as our uniqueness\
Techologies Used : Rapid API(for fethcing product information from #Amazon) and ngrok(for deployment)

### Screenshots
![Screenshot 1](https://github.com/Abhiudai12/Amazon-UNA/raw/main/una1.png)  
![Screenshot 2](https://github.com/Abhiudai12/Amazon-UNA/raw/main/una2.png)
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

