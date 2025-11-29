# Food Classifier: Hotdog vs Hamburger vs Unknown

A SetFit-based text classifier that identifies whether people are talking about hotdogs, hamburgers, or something unrelated.

## Overview

This project uses Hugging Face's SetFit (Sentence Transformer Fine-tuning) framework with the `all-MiniLM-L6-v2` base model to perform few-shot text classification with just 8 examples per class.

### Model Performance

The model achieves 56-74% confidence on test examples after training in just 3 seconds:

- "I'm craving a juicy hotdog with sauerkraut" → **hotdog** (74.4%)
- "Best burger I've ever had" → **hamburger** (70.8%)
- "The sky is blue today" → **unknown** (71.8%)
- "That hamburger place has amazing fries" → **hamburger** (73.6%)

## Setup

This project uses `uv` for dependency management.

### Prerequisites

- Python 3.13+
- uv package manager

### Installation

```bash
# Clone or navigate to the project directory
cd food

# Dependencies are managed by uv and will be installed automatically
# when you run the training script
```

## Training Data

The training dataset (`training_data.csv`) contains 24 examples:
- 8 examples about hotdogs
- 8 examples about hamburgers
- 8 examples of unrelated text (unknown)

### Data Format

```csv
text,label
I love a good hotdog with mustard and relish,hotdog
Juicy hamburgers fresh off the grill are amazing,hamburger
The weather is nice today,unknown
```

## Training the Model

Run the training script:

```bash
uv run train_model.py
```

This will:
1. Load training data from `training_data.csv`
2. Train a SetFit model using contrastive learning
3. Test the model with sample predictions
4. Save the trained model to `./food-classifier-model`

### Training Configuration

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch size: 16
- Iterations: 20 text pairs per example
- Epochs: 1
- Loss function: CosineSimilarityLoss

## Using the Model

### Python API

```python
from setfit import SetFitModel

# Load the trained model
model = SetFitModel.from_pretrained('./food-classifier-model')

# Make predictions
texts = [
    "I love hotdogs with ketchup",
    "That burger was delicious",
    "What time is the meeting?"
]

predictions = model.predict(texts)
# Output: [0, 1, 2]  # 0=hotdog, 1=hamburger, 2=unknown

# Get probabilities
probabilities = model.predict_proba(texts)
# Output: array of shape (3, 3) with probabilities for each class
```

### Label Mapping

- `0` = hotdog
- `1` = hamburger
- `2` = unknown

## Project Structure

```
food/
├── training_data.csv           # Training dataset (24 examples)
├── train_model.py              # Training script
├── food-classifier-model/      # Saved trained model
├── pyproject.toml              # Project dependencies (uv)
├── .venv/                      # Virtual environment
└── README.md                   # This file
```

## How It Works

SetFit uses a two-stage training process:

1. **Contrastive Learning**: Creates positive pairs (same class) and negative pairs (different classes) from the training data, then fine-tunes the Sentence Transformer to bring similar examples closer together in embedding space

2. **Classification Head**: Trains a simple logistic regression classifier on the fine-tuned embeddings

This approach is highly efficient:
- Trains in seconds (not hours)
- Works with just 8 examples per class
- 500x smaller than GPT-3
- No prompt engineering required

## Adding More Training Data

To improve the model, add more examples to `training_data.csv`:

```csv
text,label
Chicago-style hotdogs are the best,hotdog
I prefer sliders over regular hamburgers,hamburger
The stock market is volatile today,unknown
```

Then re-run the training script:

```bash
uv run train_model.py
```

## Technical Details

### Dependencies

- `setfit`: SetFit framework
- `pandas`: Data handling
- `datasets`: Hugging Face datasets
- `sentence-transformers`: Base embedding models
- `scikit-learn`: Classification head
- `torch`: PyTorch backend

### Model Size

- Model parameters: ~23M (all-MiniLM-L6-v2)
- Saved model size: ~90MB
- Training time: ~3 seconds on Apple Silicon

## Performance Tips

1. **More data**: Add more diverse examples to `training_data.csv`
2. **Balance classes**: Keep roughly equal examples per class
3. **Hyperparameters**: Experiment with `num_iterations` (10-50) and `batch_size` (8-32)
4. **Base model**: Try `paraphrase-mpnet-base-v2` for better quality (slower)

## References

- [SetFit Documentation](https://huggingface.co/docs/setfit)
- [SetFit Paper](https://arxiv.org/abs/2209.11055)
- [Sentence Transformers](https://www.sbert.net/)

## License

MIT
