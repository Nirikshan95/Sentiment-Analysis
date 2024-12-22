#paths
DATA_PATH="./data"
ZIP_DATA="./data/twitter_training.csv.zip"
TRAINING_DATA="./data/twitter_training.csv"
TOKENIZER_PATH="./trained models/tokenizer.pkl"
SCALER_PATH="./trained models/scaler.pkl"
MODEL_PATH="./trained models/rnn_model.keras"
MODEL_ARCHITECTURE_PATH="./results/model_architecture.json"
TRAINING_HISTORY_PATH="./results/model_history.json"

#dataset configarations
HEADER=None
NUMBER_OF_ROWS=64000
NUMBER_OF_ROWS_PER_CLASS=12000

#settigs
VOCABULARY_SIZE=70000
MAX_LENGTH=40

#model parameters
EMBEDDING_DIM=12
ACTIVATION='softmax'
OPTIMIZER='RMSProp'
LOSS='sparse_categorical_crossentropy'
METRICS=['accuracy']
EPOCHS=10
VALIDATION_SPLIT=0.2
BATCH_SIZE=10