import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_toxicity(smiles_string, model_dir="chemberta-tox21-finetuned"):
    """
    Loads the fine-tuned ChemBERTa model locally and predicts multi-task toxicity
    probabilities for a given molecular SMILES string.
    """

    model_path = os.path.abspath(model_dir)
    print(f"Loading local model and tokenizer from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)

    model.eval()

    inputs = tokenizer(
        smiles_string,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    print(f"Running inference for SMILES: '{smiles_string}'...")
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    return probabilities

if __name__ == "__main__":
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin example

    try:
        preds = predict_toxicity(test_smiles)
        print("\n--- Prediction Results ---")
        for idx, prob in enumerate(preds):
            print(f"Task {idx + 1} Probability: {prob:.4f}")
    except Exception as e:
        print(f"\nError: {e}")

