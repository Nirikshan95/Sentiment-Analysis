import config
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

#input from user
inp=input("\nEnter a sentence for sentiment analysis  (E.g: it's a good movie ): ")

# tokenizize and pad sequences in input
with open(config.TOKENIZER_PATH,'rb') as tokenize:
    tokenizer=pickle.load(tokenize)
tokenized_txt=tokenizer.texts_to_sequences([inp])
pad_text=pad_sequences(tokenized_txt,maxlen=config.MAX_LENGTH,padding='post')

#load model and predict
model=load_model(config.MODEL_PATH)
pred=model.predict(pad_text)
classes=['Negative','Positive','Neutral','Irrelevant']
print(classes[np.argmax(pred)])