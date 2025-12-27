#!/usr/bin/env python3
"""
Weave evaluation script for food classifier SetFit model
Evaluates hotdog/hamburger/unknown classification with W&B Weave
"""

import asyncio
import pandas as pd
import weave
from setfit import SetFitModel


# Label mapping (same as training)
LABEL_MAP = {'hotdog': 0, 'hamburger': 1, 'unknown': 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# Load model once at module level
MODEL = SetFitModel.from_pretrained("./food-classifier-model")


@weave.op()
def predict_food(text: str) -> dict:
    """Predict food category for a single text input"""
    # Get prediction and probabilities
    pred = MODEL.predict([text])[0]
    probs = MODEL.predict_proba([text])[0]

    # Convert prediction to int
    pred_id = int(pred) if hasattr(pred, 'item') else pred
    predicted_label = ID_TO_LABEL[pred_id]

    return {
        "prediction": predicted_label,
        "prediction_id": pred_id,
        "probabilities": {
            "hotdog": float(probs[0]),
            "hamburger": float(probs[1]),
            "unknown": float(probs[2])
        },
        "confidence": float(probs[pred_id])
    }


@weave.op()
def accuracy_scorer(label: str, output: dict) -> dict:
    """Score prediction accuracy"""
    correct = label == output["prediction"]
    return {"correct": correct}


@weave.op()
def confidence_scorer(label: str, output: dict) -> dict:
    """Score prediction confidence"""
    return {"confidence": output["confidence"]}


@weave.op()
def multi_metric_scorer(label: str, output: dict) -> dict:
    """Combined scorer for accuracy and confidence"""
    correct = label == output["prediction"]
    confidence = output["confidence"]

    # High-confidence correct predictions are best
    quality_score = confidence if correct else 0.0

    return {
        "correct": correct,
        "confidence": confidence,
        "quality_score": quality_score
    }


def load_eval_data(csv_path: str) -> list[dict]:
    """Load evaluation data from CSV"""
    df = pd.read_csv(csv_path)

    # Convert to list of dicts for weave
    eval_data = df.to_dict('records')

    print(f"Loaded {len(eval_data)} evaluation examples")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    return eval_data


async def main():
    """Run evaluation with W&B Weave"""

    # Initialize Weave
    print("Initializing W&B Weave...")
    weave.init("food-classifier")

    # Load evaluation data
    print("\nLoading evaluation data...")
    eval_data = load_eval_data("eval.csv")

    # Create evaluation
    print("\nCreating evaluation...")
    evaluation = weave.Evaluation(
        dataset=eval_data,
        scorers=[accuracy_scorer, confidence_scorer, multi_metric_scorer],
        name="food-classifier-eval"
    )

    # Run evaluation
    print("\nRunning evaluation...")
    print("=" * 60)
    results = await evaluation.evaluate(predict_food)

    # Print results summary
    print("\n" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nEvaluation dataset: eval.csv ({len(eval_data)} examples)")
    print(f"\nResults: {results}")

    print("\n✓ Evaluation complete!")
    print("\nView detailed results in W&B Weave dashboard:")
    print("https://wandb.ai/standdio/food-classifier/weave")


if __name__ == "__main__":
    asyncio.run(main())
