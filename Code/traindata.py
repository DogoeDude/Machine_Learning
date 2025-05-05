import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
TRAINED_DATA_DIR = os.path.join(BASE_DIR, 'Data')  # Store trained models in Data folder

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAINED_DATA_DIR, exist_ok=True)

# Load the Excel file - now looking for it in the Machine_Learning directory
data_file = os.path.join(BASE_DIR, "TranscriptionData_with_Intents_Objects - Final.xlsx")

# Check if file exists
if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file not found at: {data_file}. Please ensure the Excel file is in the Machine_Learning directory.")

try:
    df = pd.read_excel(data_file, sheet_name="TranscriptionData_with_Intents_")
    df = df[['Transcription', 'Intent', 'Object']].dropna()
except Exception as e:
    raise Exception(f"Error reading Excel file: {str(e)}")

# Filter out noisy data
df = df[df['Intent'] != 'Unknown']  # Remove Unknown intent
df = df[df['Object'] != 'Unknown']  # Remove Unknown object

# Print updated statistics
print("\nAfter filtering:")
print(f"Total number of samples: {len(df)}")
print("\nIntent Distribution:")
intent_counts = df['Intent'].value_counts()
print(intent_counts)
print("\nObject Distribution:")
object_counts = df['Object'].value_counts()
print(object_counts)

# Check for class imbalance
print("\nClass Imbalance Analysis:")
print("Intent class imbalance ratio:", intent_counts.max() / intent_counts.min())
print("Object class imbalance ratio:", object_counts.max() / object_counts.min())

# Print some example samples for each intent
print("\nExample Samples by Intent:")
for intent in df['Intent'].unique():
    print(f"\nIntent: {intent}")
    samples = df[df['Intent'] == intent].head(2)
    for _, row in samples.iterrows():
        print(f"Text: {row['Transcription']}")
        print(f"Object: {row['Object']}")
        print()

# Print unique intents and objects for debugging
print("Unique Intents:", df['Intent'].unique())
print("Unique Objects:", df['Object'].unique())

# Encode text and labels
texts = df['Transcription'].astype(str)
intent_encoder = LabelEncoder()
object_encoder = LabelEncoder()

intent_encoded = intent_encoder.fit_transform(df['Intent'])
object_encoded = object_encoder.fit_transform(df['Object'])

# Print the mapping for debugging
print("\nIntent mapping:", dict(zip(intent_encoder.classes_, intent_encoder.transform(intent_encoder.classes_))))
print("Object mapping:", dict(zip(object_encoder.classes_, object_encoder.transform(object_encoder.classes_))))

intent_categorical = to_categorical(intent_encoded)
object_categorical = to_categorical(object_encoded)

# Tokenize text
vocab_size = 10000
max_len = 25
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# Print vocabulary information
print("\nVocabulary Information:")
print(f"Vocabulary size: {len(tokenizer.word_index)}")
print("Most common words:", list(tokenizer.word_index.items())[:10])

# Print example of tokenization
print("\nExample of tokenization:")
for i in range(min(3, len(texts))):
    print(f"Original text: {texts[i]}")
    print(f"Tokenized: {sequences[i]}")
    print(f"Padded: {padded[i]}")
    print()

# Train/test split with stratification
X_train, X_val, y_intent_train, y_intent_val, y_object_train, y_object_val = train_test_split(
    padded, 
    intent_categorical, 
    object_categorical, 
    test_size=0.2, 
    random_state=42,
    stratify=intent_encoded  # Ensure balanced distribution of intents
)

print("\nTraining set size:", len(X_train))
print("Validation set size:", len(X_val))

# Print class distribution in training and validation sets
print("\nTraining set class distribution:")
print("Intent distribution:", np.sum(y_intent_train, axis=0))
print("Object distribution:", np.sum(y_object_train, axis=0))

print("\nValidation set class distribution:")
print("Intent distribution:", np.sum(y_intent_val, axis=0))
print("Object distribution:", np.sum(y_object_val, axis=0))

# Verify that we have samples for all classes in both sets
print("\nVerifying class coverage:")
print("Unique intents in training:", np.unique(np.argmax(y_intent_train, axis=1)))
print("Unique intents in validation:", np.unique(np.argmax(y_intent_val, axis=1)))
print("Unique objects in training:", np.unique(np.argmax(y_object_train, axis=1)))
print("Unique objects in validation:", np.unique(np.argmax(y_object_val, axis=1)))

# Calculate class weights for intent
intent_class_weights = {}
for i, count in enumerate(np.sum(y_intent_train, axis=0)):
    intent_class_weights[i] = len(y_intent_train) / (len(intent_encoder.classes_) * count)

# Calculate class weights for object
object_class_weights = {}
for i, count in enumerate(np.sum(y_object_train, axis=0)):
    object_class_weights[i] = len(y_object_train) / (len(object_encoder.classes_) * count)

# Intent model
intent_model = tf.keras.Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 300),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(intent_categorical.shape[1], activation='softmax')
])

intent_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Object model
object_model = tf.keras.Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 300),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128, return_sequences=True)),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(object_categorical.shape[1], activation='softmax')
])

object_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Add early stopping and model checkpoint
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6
)

# Train intent model
print("\nTraining Intent Model:")
intent_history = intent_model.fit(
    X_train,
    y_intent_train,
    validation_data=(X_val, y_intent_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    class_weight=intent_class_weights
)

# Train object model
print("\nTraining Object Model:")
object_history = object_model.fit(
    X_train,
    y_object_train,
    validation_data=(X_val, y_object_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    class_weight=object_class_weights
)

# Save models
intent_model.save(os.path.join(TRAINED_DATA_DIR, "intent_model.keras"))
object_model.save(os.path.join(TRAINED_DATA_DIR, "object_model.keras"))

# Save tokenizer
with open(os.path.join(TRAINED_DATA_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

# Save intent encoder
with open(os.path.join(TRAINED_DATA_DIR, "intent_encoder.pkl"), "wb") as f:
    pickle.dump(intent_encoder, f)

# Save object encoder
with open(os.path.join(TRAINED_DATA_DIR, "object_encoder.pkl"), "wb") as f:
    pickle.dump(object_encoder, f)

print(f"\nModels and encoders saved in: {TRAINED_DATA_DIR}")