# Intent and Object Recognition Model

This project implements a dual-task neural network model for recognizing intents and objects from natural language commands. The model is trained to identify both the action (intent) and the target (object) from user commands.

## Model Architecture

The system uses two separate models:
1. **Intent Model**: Classifies the action/command
2. **Object Model**: Classifies the target object

Both models share the same architecture:
- Input Layer
- Embedding Layer (300 dimensions)
- Spatial Dropout (0.2)
- Bidirectional LSTM (128 units)
- Global Max Pooling
- Dense Layer (128 units, ReLU)
- Dropout (0.3)
- Dense Layer (64 units, ReLU)
- Dropout (0.3)
- Output Layer (Softmax)

## Training Process

### Data Preparation
1. Data is loaded from `TranscriptionData_with_Intents_Objects - Final.xlsx`
2. Preprocessing steps:
   - Removes rows with 'Unknown' intents or objects
   - Converts text to lowercase
   - Tokenizes text with vocabulary size of 10,000
   - Pads sequences to max length of 25

### Data Split
- Training set: 80%
- Validation set: 20%
- Stratified split to maintain class distribution

### Training Parameters
- Optimizer: Adam (learning rate: 0.001)
- Loss: Categorical Crossentropy
- Batch size: 32
- Maximum epochs: 30
- Early stopping with patience of 5
- Learning rate reduction on plateau
- Class weights to handle imbalance

## Available Commands and Objects

### Intents (Actions)
1. Activate
2. Bring
3. Change Language
4. Deactivate
5. Decrease
6. Increase

### Objects
1. Languages:
   - Chinese
   - English
   - German
   - Korean
2. Devices/Items:
   - heat
   - juice
   - lamp
   - lights
   - music
   - newspaper
   - shoes
   - socks
   - volume
3. Special:
   - none

## How to Cleanse/Retrain the Model

1. **Data Cleansing**:
   - Review and clean the Excel file `TranscriptionData_with_Intents_Objects - Final.xlsx`
   - Remove any incorrect or ambiguous entries
   - Ensure consistent formatting
   - Add more examples for underrepresented classes

2. **Retraining Process**:
   ```bash
   # 1. Activate your virtual environment
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows

   # 2. Run the training script
   python traindata.py
   ```

3. **Model Evaluation**:
   - The script will print training statistics
   - Check class distribution in training and validation sets
   - Monitor for any class imbalance
   - Verify that all classes are represented in both sets

4. **Saving Models**:
   The script automatically saves:
   - `intent_model.keras`: Intent classification model
   - `object_model.keras`: Object classification model
   - `tokenizer.pkl`: Text tokenizer
   - `intent_encoder.pkl`: Intent label encoder
   - `object_encoder.pkl`: Object label encoder

## Testing the Model

Use `test_interference.py` to test the model with new commands:
```python
python test_interference.py
```

The script will show:
- Tokenization process
- Top 3 predictions for both intent and object
- Confidence scores
- Final predictions

## Requirements

- Python 3.x
- TensorFlow
- pandas
- scikit-learn
- numpy
- openpyxl (for Excel file handling)

## Notes

- The model uses class weights to handle imbalanced data
- Early stopping prevents overfitting
- Learning rate reduction helps in fine-tuning
- The model architecture is optimized for short commands
- Maximum sequence length is 25 tokens 