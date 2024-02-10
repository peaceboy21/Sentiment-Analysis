from sentimentanalysis.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
import torch

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer_config()
        tokenizer_path = "C:/Users/91787/Programming/Projects/Sentiment-Analysis_Roberta/artifacts/model_trainer/tokenizer"
        model_path = "C:/Users/91787/Programming/Projects/Sentiment-Analysis_Roberta/artifacts/model_trainer/pegasus-samsum-model"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.sia = SentimentIntensityAnalyzer()
        self.device = torch.device("cpu")  # or "cuda" if available

    def polarity_scores_roberta(self, example):
        encoded_text = self.tokenizer(example, return_tensors='pt')
        encoded_text = encoded_text.to(self.device)
        
        output = self.model(**encoded_text)
        scores = output.logits.detach().cpu().numpy()
        scores = softmax(scores, axis=1)  # Ensure softmax is applied along the correct axis
        
        scores_dict = {
            'roberta_neg' : scores[0, 0],  # Adjust indexing
            'roberta_neu' : scores[0, 1],  # Adjust indexing
            'roberta_pos' : scores[0, 2],  # Adjust indexing
        }
        return scores_dict 

    def sentiment_analysis(self, text):
        # Test VADER Sentiment Analysis
        sample_vader_result = self.sia.polarity_scores(text)

        # Test Roberta Pretrained Model
        sample_roberta_result = self.polarity_scores_roberta(text)

        # Determine overall sentiment
        threshold = 0.1  # Define a small threshold for neutrality

        if abs(sample_vader_result['compound']) <= threshold:
            overall_sentiment = "Neutral"
            confidence_percent = 50  # Set confidence to 50% for neutral sentiment
        elif sample_vader_result['compound'] > 0 and sample_roberta_result['roberta_pos'] > sample_roberta_result['roberta_neg']:
            overall_sentiment = "Positive"
            confidence_percent = sample_vader_result['pos'] * 100
        elif sample_vader_result['compound'] < 0 and sample_roberta_result['roberta_pos'] < sample_roberta_result['roberta_neg']:
            overall_sentiment = "Negative"
            confidence_percent = sample_vader_result['neg'] * 100


        return {
            "text": text,
            "sentiment": overall_sentiment,
            "confidence_percent": confidence_percent
        }

# Instantiate PredictionPipeline
pipeline_instance = PredictionPipeline()

# Sample text for testing sentiment
sample_text = "it feels amazing today the sky and the birds"

# Call the sentiment_analysis function
result = pipeline_instance.sentiment_analysis(sample_text)
print(result)
