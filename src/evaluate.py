import config,preprocessing
from tensorflow.keras.models import load_model

review,sentiment=preprocessing.data()
model=load_model(config.MODEL_PATH)
model.evaluate(review,sentiment)