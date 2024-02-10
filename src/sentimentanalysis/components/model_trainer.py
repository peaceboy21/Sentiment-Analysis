import os
import pandas as pd
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.special import softmax
from tqdm.notebook import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentimentanalysis.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Read in data
        df = pd.read_csv("C:\\Users\\91787\\Programming\\Projects\\Sentiment-Analysis_Roberta\\data\\Reviews.csv")
        df = df.head(500)

        # Step 1: VADER Sentiment Scoring
        sia = SentimentIntensityAnalyzer()

        # Run the polarity score on the entire dataset
        res = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            text = row['Text']
            myid = row['Id']
            res[myid] = sia.polarity_scores(text)

        vaders = pd.DataFrame(res).T
        vaders = vaders.reset_index().rename(columns={'index': 'Id'})
        vaders = vaders.merge(df, how='left')

        # Step 3: Roberta Pretrained Model
        MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.to(device)

        def polarity_scores_roberta(example):
            encoded_text = tokenizer(example, return_tensors='pt')
            encoded_text = encoded_text.to(device)
            
            output = model(**encoded_text)
            scores = output.logits.detach().cpu().numpy()
            scores = softmax(scores, axis=1)  # Ensure softmax is applied along the correct axis
            
            scores_dict = {
                'roberta_neg' : scores[0, 0],  # Adjust indexing
                'roberta_neu' : scores[0, 1],  # Adjust indexing
                'roberta_pos' : scores[0, 2],  # Adjust indexing
            }
            return scores_dict

        res = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            try:
                text = row['Text']
                myid = row['Id']
                vader_result = sia.polarity_scores(text)
                vader_result_rename = {}
                for key, value in vader_result.items():
                    vader_result_rename[f"vader_{key}"] = value
                roberta_result = polarity_scores_roberta(text)
                both = {**vader_result_rename, **roberta_result}
                res[myid] = both
            except RuntimeError:
                print(f'Broke for id {myid}')

        results_df = pd.DataFrame(res).T
        results_df = results_df.reset_index().rename(columns={'index': 'Id'})
        results_df = results_df.merge(df, how='left')

        # Save the tokenizer and model
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
        model.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
