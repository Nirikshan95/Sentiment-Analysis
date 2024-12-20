import config,preprocessing
import os,json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout

a,b=preprocessing.data()

#build model
model=Sequential([
    Embedding(config.VOCABULARY_SIZE,config.EMBEDDING_DIM,input_length=config.MAX_LENGTH),
    LSTM(30,activation='tanh'),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dense(4,activation=config.ACTIVATION)
]
)

#compile the model
model.compile(optimizer=config.OPTIMIZER,loss=config.LOSS,metrics=config.METRICS)



#train the model
history=model.fit(a,b,validation_split=config.VALIDATION_SPLIT,epochs=config.EPOCHS,batch_size=config.BATCH_SIZE)

#print model summary
#model.summary()

#save model
os.makedirs('./trained models',exist_ok=True)
model.save(config.MODEL_PATH)

#save model architecture
os.makedirs('./results',exist_ok=True)
with open(config.MODEL_ARCHITECTURE_PATH,'w') as arc:
    json.dump(model.to_json(),arc)

#training history
os.makedirs('./results',exist_ok=True)
with open(config.TRAINING_HISTORY_PATH,'w') as hist:
    json.dump(history.history,hist)