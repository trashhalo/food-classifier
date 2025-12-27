# Food Classifier: Hotdog vs Hamburger vs Unknown

A SetFit-based text classifier that identifies whether people are talking about hotdogs, hamburgers, or something unrelated.

## Overview

This project uses Hugging Face's SetFit (Sentence Transformer Fine-tuning) framework with the `all-MiniLM-L6-v2` base model to perform few-shot text classification with just 8 examples per class.

**Features:**
- 🚀 Fast training (3 seconds on Apple Silicon)
- 📊 W&B Weave integration for evaluation tracking
- 🎯 100% accuracy on test set with 72% average confidence
- 💾 Lightweight model (~90MB)

### Model Performance

The model achieves 56-74% confidence on test examples after training in just 3 seconds:

- "I'm craving a juicy hotdog with sauerkraut" → **hotdog** (74.4%)
- "Best burger I've ever had" → **hamburger** (70.8%)
- "The sky is blue today" → **unknown** (71.8%)
- "That hamburger place has amazing fries" → **hamburger** (73.6%)

## Quick Start

```bash
# Train the model
uv run train_model.py

# Run evaluation with W&B Weave
uv run eval.py
```

## Setup

This project uses `uv` for dependency management.

### Prerequisites

- Python 3.13+
- uv package manager
- W&B account (for evaluation tracking)

### Installation

```bash
# Clone or navigate to the project directory
cd food

# Dependencies are managed by uv and will be installed automatically
# when you run the training script

# (Optional) Set up W&B API key for evaluation
# Create mise.local.toml with your WANDB_API_KEY
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

## Evaluation with W&B Weave

The project includes comprehensive evaluation using Weights & Biases Weave for tracking model performance.

### Setup W&B API Key

1. Create a `mise.local.toml` file (already gitignored):
```toml
[env]
WANDB_API_KEY = "your-api-key-here"
```

2. Trust the mise configuration:
```bash
mise trust
```

### Running Evaluation

```bash
uv run eval.py
```

This will:
1. Load evaluation data from `eval.csv` (14 test examples)
2. Run predictions on all examples
3. Compute accuracy, confidence, and quality metrics
4. Upload results to W&B Weave dashboard

### Evaluation Metrics

The evaluation tracks:
- **Accuracy**: Percentage of correct predictions
- **Confidence**: Average model confidence scores
- **Quality Score**: Confidence-weighted accuracy
- **Model Latency**: Average prediction time

### Viewing Results

After running evaluation, view detailed results at:
```
https://wandb.ai/your-username/food-classifier/weave
```

The Weave dashboard provides:
- Per-example predictions and scores
- Aggregated metrics across the dataset
- Probability distributions
- Latency measurements

### Current Performance

On the evaluation dataset (14 examples):
- **Accuracy**: 100% (14/14 correct)
- **Average Confidence**: 72%
- **Mean Latency**: 2.6 seconds

### Creating Custom Evaluation Data

Add new test examples to `eval.csv` using the same format as training data:

```csv
text,label
I'm craving a hotdog with sauerkraut right now,hotdog
The burger had too much mayo on it,hamburger
The sky is beautiful this evening,unknown
```

The evaluation script automatically:
- Loads all examples from the CSV
- Runs predictions using the trained model
- Computes metrics using W&B Weave scorers
- Logs everything to the Weave dashboard for analysis

## Project Structure

```
food/
├── training_data.csv           # Training dataset (24 examples)
├── eval.csv                    # Evaluation dataset (14 examples)
├── train_model.py              # Training script
├── eval.py                     # W&B Weave evaluation script
├── food-classifier-model/      # Saved trained model
├── mise.local.toml             # Local config with API keys (gitignored)
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
- `weave`: W&B Weave for evaluation tracking
- `wandb`: Weights & Biases integration

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
