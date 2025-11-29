#!/usr/bin/env python3
"""
SetFit training script for hotdog/hamburger/unknown classification
Uses all-MiniLM-L6-v2 base model
"""

import pandas as pd
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

def load_data(csv_path):
    """Load training data from CSV and convert to HuggingFace Dataset"""
    df = pd.read_csv(csv_path)

    # Convert labels to integers (required by SetFit)
    label_map = {'hotdog': 0, 'hamburger': 1, 'unknown': 2}
    df['label'] = df['label'].map(label_map)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)

    print(f"Loaded {len(dataset)} training examples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return dataset, label_map

def train_model(train_dataset, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Train SetFit model"""

    print(f"\nLoading base model: {model_name}")
    model = SetFitModel.from_pretrained(model_name)

    print("\nCreating trainer...")
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_iterations=20,  # Number of text pairs for contrastive learning
        num_epochs=1,       # Epochs for contrastive learning phase
    )

    print("\nTraining model...")
    trainer.train()

    return trainer.model

def test_model(model, label_map):
    """Test the trained model with sample predictions"""

    # Reverse label map for display
    id_to_label = {v: k for k, v in label_map.items()}

    test_texts = [
        "I'm craving a juicy hotdog with sauerkraut",
        "Best burger I've ever had",
        "The sky is blue today",
        "Do you like hotdogs or hamburgers better",
        "That hamburger place has amazing fries",
        "I need to buy groceries",
    ]

    print("\n" + "="*60)
    print("Testing model predictions:")
    print("="*60)

    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)

    for text, pred, probs in zip(test_texts, predictions, probabilities):
        # Convert tensor to int if necessary
        pred_id = int(pred) if hasattr(pred, 'item') else pred
        predicted_label = id_to_label[pred_id]
        confidence = probs[pred_id] * 100
        print(f"\nText: {text}")
        print(f"Prediction: {predicted_label} (confidence: {confidence:.1f}%)")
        print(f"All probabilities: {dict(zip(['hotdog', 'hamburger', 'unknown'], [f'{p*100:.1f}%' for p in probs]))}")

def main():
    # Load training data
    print("Loading training data...")
    train_dataset, label_map = load_data("training_data.csv")

    # Train model
    model = train_model(train_dataset)

    # Test model
    test_model(model, label_map)

    # Save model
    output_dir = "./food-classifier-model"
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)

    print("\n✓ Training complete!")
    print(f"Model saved to: {output_dir}")
    print("\nTo use the model:")
    print("  from setfit import SetFitModel")
    print(f"  model = SetFitModel.from_pretrained('{output_dir}')")
    print("  predictions = model.predict(['your text here'])")

if __name__ == "__main__":
    main()
