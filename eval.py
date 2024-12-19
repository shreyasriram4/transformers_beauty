import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import ExplicitBTBERT
from preprocessing import load_and_preprocess_data

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model and calculate predictions.

    Args:
        model: The model to evaluate.
        val_loader: DataLoader for validation data.
        device: Device to use for evaluation.

    Returns:
        all_preds, all_labels: Predictions and ground-truth labels.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_preds, all_labels

def calculate_metrics(all_preds, all_labels, attribute_names):
    """
    Calculate precision, recall, F1-score, and support for each attribute.

    Args:
        all_preds: Model predictions.
        all_labels: Ground-truth labels.
        attribute_names: List of attribute names.

    Returns:
        metrics_by_class: Dictionary of metrics by class.
    """
    # Metrics by class
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    metrics_by_class = {
        attr: {
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f, 4),
            "Support": int(s)
        }
        for attr, p, r, f, s in zip(attribute_names, precision, recall, f1, support)
    }

    # Overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro")
    overall_metrics = {
        "Precision (Macro Avg)": round(overall_precision, 4),
        "Recall (Macro Avg)": round(overall_recall, 4),
        "F1-Score (Macro Avg)": round(overall_f1, 4)
    }

    return metrics_by_class, overall_metrics

def plot_confusion_matrices(all_preds, all_labels, attribute_names):
    """
    Plot confusion matrices for each attribute.

    Args:
        all_preds: Predictions for all attributes.
        all_labels: Ground-truth labels for all attributes.
        attribute_names: List of attribute names.
    """
    for i, attribute in enumerate(attribute_names):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Present", "Present"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {attribute}")
        plt.show()

def display_classification_report(all_preds, all_labels, attribute_names):
    """
    Display classification report for each attribute.

    Args:
        all_preds: Model predictions.
        all_labels: Ground-truth labels.
        attribute_names: List of attribute names.
    """
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=attribute_names, zero_division=0)
    print(report)

if __name__ == "__main__":
    # Parameters
    attributes = ["acne", "combination skin", "dark circles", "dark spots", "fine lines and wrinkles",
                  "hydration", "normal skin", "oily skin", "pores", "redness", "sagging",
                  "sensitive skin", "uneven texture"]
    file_path = "all_attributes_modeling.csv"

    # Load model and data
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    train_loader, val_loader = load_and_preprocess_data(file_path, attributes, tokenizer)
    model = ExplicitBTBERT(model_name="BAAI/bge-base-en-v1.5", num_labels=len(attributes))
    model.load_state_dict(torch.load("model.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate model
    all_preds, all_labels = evaluate_model(model, val_loader, device)

    # Calculate metrics
    metrics_by_class, overall_metrics = calculate_metrics(all_preds, all_labels, attributes)
    print("\nOverall Metrics:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")
    
    print("\nMetrics by Class:")
    for attr, metrics in metrics_by_class.items():
        print(f"\nAttribute: {attr}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    # Display classification report
    display_classification_report(all_preds, all_labels, attributes)

    # Plot confusion matrices
    plot_confusion_matrices(all_preds, all_labels, attributes)
