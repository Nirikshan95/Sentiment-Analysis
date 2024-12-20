import pandas as pd
import numpy as np
import pickle
import zipfile
import os
import config
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    def get_data(self):
        if not os.path.exists(config.ZIP_DATA):
            os.makedirs(os.path.dirname(config.DATA_PATH),exist_ok=True)
            print('insert the zipfile in in the data folder ')
            return
        
        #unzip
        if not os.path.exists(config.TRAINING_DATA):
            with zipfile.ZipFile(config.ZIP_DATA,'r') as zip_ref:
                zip_ref.extractall(config.DATA_PATH)
                
        #read as pandas DataFrame
        df=pd.read_csv(config.TRAINING_DATA,header=config.HEADER,nrows=config.NUMBER_OF_ROWS)

        return df
                
    def preprocess(self,df):
        
        #drops the null values
        df.dropna(inplace=True)
        
        df_p=df[df[2]=='Positive'].tail(config.NUMBER_OF_ROWS_PER_CLASS)
        df_neu=df[df[2]=='Neutral'].tail(config.NUMBER_OF_ROWS_PER_CLASS)
        df_neg=df[df[2]=='Negative'].tail(config.NUMBER_OF_ROWS_PER_CLASS)
        df_irr=df[df[2]=='Irrelevant'].tail(config.NUMBER_OF_ROWS_PER_CLASS)
        
        concatinated_data=pd.concat([df_p,df_neu,df_neg,df_irr],axis=0)
        #shufflig
        shuffled_df=shuffle(concatinated_data)
        shuffled_df.reset_index(drop=True,inplace=True)
        
        #prepare the data
        X=shuffled_df[3]
        print(X)
        y=shuffled_df[2].map({'Negative':0,'Positive':1,'Neutral':2,'Irrelevant':3}).values
        
        #tokenization
        tokenizer=Tokenizer(num_words=config.VOCABULARY_SIZE,oov_token='<oov>')
        tokenizer.fit_on_texts(X)
        tokenized=tokenizer.texts_to_sequences(X)
        import matplotlib.pyplot as plt
        lengths = [len(seq) for seq in tokenized]
        plt.hist(lengths, bins=50)
        plt.show()

        
        
        #save the tokenizer
        os.makedirs('./trained models',exist_ok=True)
        with open(config.TOKENIZER_PATH,'wb') as f:
            pickle.dump(tokenizer,f)        
        
        #padding
        padded_X=pad_sequences(tokenized,maxlen=config.MAX_LENGTH,padding='post')
        scaler=MinMaxScaler()
        scaled_X=scaler.fit_transform(padded_X)
        return scaled_X,y

def data():
    dt=DataLoader()
    processed_data=dt.get_data()
    X,y=dt.preprocess(processed_data)
    return X,y
    
if __name__=="__main__":
    review,sentiment=data()
