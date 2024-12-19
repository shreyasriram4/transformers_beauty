import torch
import re
from collections import defaultdict
from lime.lime_text import LimeTextExplainer

def extract_ingredients(text, ingredients_list):
    """
    Extract explicitly matched ingredients from a given text.
    
    Args:
        text: Input text (e.g., product description or metadata).
        ingredients_list: List of known ingredient names for matching.
    
    Returns:
        A comma-separated string of matched ingredients, or a default message if no matches found.
    """
    matched_ingredients = []
    for ing in ingredients_list:
        if re.search(rf'\b{re.escape(ing)}\b', text, re.IGNORECASE):
            matched_ingredients.append(ing)
    return ", ".join(matched_ingredients) if matched_ingredients else "No Ingredients Found"

def predict_function_lime(texts, tokenizer, model, device):
    """
    Define the prediction function for LIME, wrapping the model inference process.
    
    Args:
        texts: List of input texts to predict.
    
    Returns:
        Predicted probabilities for each label across the texts.
    """
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits, _ = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.sigmoid(logits).cpu().numpy()  # Convert logits to probabilities
    return probs

def aggregate_lime_explanations(texts, labels, label_index, ingredients_list, num_samples=50):
    """
    Aggregate LIME explanations for a specific target label across multiple examples.
    
    Args:
        model: The trained classification model.
        tokenizer: Tokenizer used for input processing.
        texts: List of input texts (e.g., validation set).
        labels: Corresponding labels for the texts.
        label_index: Index of the target label for which explanations are computed.
        ingredients_list: List of known ingredient names.
        num_samples: Number of perturbations used by LIME for each explanation.
    
    Returns:
        Sorted list of aggregated token importance scores for the target label.
    """
    attributes = ["acne", "combination skin", "dark circles", "dark spots", "fine lines and wrinkles",
              "hydration", "normal skin", "oily skin", "pores", "redness", "sagging",
              "sensitive skin", "uneven texture"]
    explainer = LimeTextExplainer(class_names=attributes)
    aggregated_importance = defaultdict(float)
    example_count = 0

    for i, (text, label) in enumerate(zip(texts, labels)):
        if label[label_index] == 1:  # Only consider texts where the target label is active
            cleaned_text = extract_ingredients(text, ingredients_list)  # Focus on ingredients
            if cleaned_text == "No Ingredients Found":
                continue

            explanation = explainer.explain_instance(
                cleaned_text, predict_function_lime, num_features=20, num_samples=num_samples, labels=[label_index]
            )

            for token, weight in explanation.as_list(label_index):
                aggregated_importance[token] += weight

            example_count += 1
            print(f"Processed example {example_count}: {cleaned_text}")

    for token in aggregated_importance:
        aggregated_importance[token] /= example_count  # Normalize importance by example count

    return sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
