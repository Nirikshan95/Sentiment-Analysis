#paths
DATA_PATH="./data"
ZIP_DATA="./data/twitter_training.csv.zip"
TRAINING_DATA="./data/twitter_training.csv"
TOKENIZER_PATH="./trained models/tokenizer.pkl"
MODEL_PATH="./trained models/Simple_rnn_model.keras"
MODEL_ARCHITECTURE_PATH="./results/model_architecture.json"
TRAINING_HISTORY_PATH="./results/model_history.json"

#dataset configarations
HEADER=None
NUMBER_OF_ROWS=15000
NUMBER_OF_ROWS_PER_CLASS=2000

#settigs
VOCABULARY_SIZE=5000
MAX_LENGTH=50

#model parameters
EMBEDDING_DIM=12
ACTIVATION='softmax'
OPTIMIZER='adam'
LOSS='sparse_categorical_crossentropy'
METRICS=['accuracy']
EPOCHS=10
VALIDATION_SPLIT=0.2
BATCH_SIZE=10