from tensorflow.keras.models import load_model
import pickle
import numpy as np
import tensorflow as tf
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_DATA_DIR = os.path.join(BASE_DIR, 'Data')  # Models are stored in Data folder

# Check if required files exist
required_files = [
    "intent_model.keras",
    "object_model.keras",
    "tokenizer.pkl",
    "intent_encoder.pkl",
    "object_encoder.pkl"
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(TRAINED_DATA_DIR, f))]
if missing_files:
    raise FileNotFoundError(f"Missing required files in {TRAINED_DATA_DIR}: {', '.join(missing_files)}")

# Load everything
print("Loading models and encoders...")
intent_model = load_model(os.path.join(TRAINED_DATA_DIR, "intent_model.keras"))
object_model = load_model(os.path.join(TRAINED_DATA_DIR, "object_model.keras"))

with open(os.path.join(TRAINED_DATA_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
with open(os.path.join(TRAINED_DATA_DIR, "intent_encoder.pkl"), "rb") as f:
    intent_encoder = pickle.load(f)
with open(os.path.join(TRAINED_DATA_DIR, "object_encoder.pkl"), "rb") as f:
    object_encoder = pickle.load(f)

# Print available classes for debugging
print("Available intents:", intent_encoder.classes_)
print("Available objects:", object_encoder.classes_)

MAX_SEQUENCE_LENGTH = 25

# Preprocessing
def predict_intent_and_object(text):
    print(f"\nProcessing text: '{text}'")
    
    # Preprocess text
    text = text.lower().strip()
    print(f"Preprocessed text: '{text}'")
    
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([text])
    print("Tokenized sequence:", seq)
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    print("Padded sequence:", padded)
    
    # Get predictions
    intent_pred = intent_model.predict(padded)
    object_pred = object_model.predict(padded)
    
    # Get top 3 predictions for each
    intent_top3 = np.argsort(intent_pred[0])[-3:][::-1]
    object_top3 = np.argsort(object_pred[0])[-3:][::-1]
    
    print("\nTop 3 Intent Predictions:")
    for idx in intent_top3:
        print(f"{intent_encoder.classes_[idx]}: {intent_pred[0][idx]:.4f}")
    
    print("\nTop 3 Object Predictions:")
    for idx in object_top3:
        print(f"{object_encoder.classes_[idx]}: {object_pred[0][idx]:.4f}")
    
    # Get the predicted classes
    intent_idx = np.argmax(intent_pred)
    object_idx = np.argmax(object_pred)
    
    # Get confidence scores
    intent_confidence = intent_pred[0][intent_idx]
    object_confidence = object_pred[0][object_idx]
    
    # Convert to labels
    intent = intent_encoder.inverse_transform([intent_idx])
    object_ = object_encoder.inverse_transform([object_idx])
    
    print(f"\nFinal Predictions:")
    print(f"Intent: {intent[0]} (confidence: {intent_confidence:.4f})")
    print(f"Object: {object_[0]} (confidence: {object_confidence:.4f})")
    
    # Check if predictions are uncertain
    if intent_confidence < 0.5 or object_confidence < 0.5:
        print("\nWarning: Low confidence in predictions!")
    
    return intent[0], object_[0]

# Test with multiple examples
test_examples = [
    "change language to chinese",
    "change language to german",
    "activate lights",
    "deactivate music",
    "increase volume",
    "decrease heat",
    "bring newspaper",
    "bring juice"
]

for test_text in test_examples:
    print("\n" + "="*50)
    print("Testing with:", test_text)
    intent, obj = predict_intent_and_object(test_text)
    print(f"Predicted Intent: {intent}")
    print(f"Predicted Object: {obj}")
