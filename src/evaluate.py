import config,preprocessing
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

df=pd.read_csv(config.TRAINING_DATA,skiprows=70000,nrows=3000,header=None)
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

review=df[3]
sentiment=df[2].map({'Negative':0,'Positive':1,'Neutral':2,'Irrelevant':3}).values

# tokenize and pad sequences
with open(config.TOKENIZER_PATH,'rb') as tokenize:
    tokenizer=pickle.load(tokenize)
tokenized_txts=tokenizer.texts_to_sequences(review)
padded_sqns=pad_sequences(tokenized_txts,maxlen=config.MAX_LENGTH,padding='post')

#load model and evaluate
model=load_model(config.MODEL_PATH)
model.evaluate(padded_sqns,sentiment)