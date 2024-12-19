import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel
from preprocessing import load_and_preprocess_data

class ExplicitBTBERT(nn.Module):
    """
    Multi-label classification model with a BERT backbone.
    """
    def __init__(self, model_name, num_labels):
        super(ExplicitBTBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

def train_classifier(model, train_loader, optimizer, criterion, device, epochs=4):
    """
    Train the classifier head of the model.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to use for training.
        epochs: Number of training epochs.
    """
    model.bert.requires_grad_(False)  # Freeze backbone
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    # Parameters
    attributes = ["acne", "combination skin", "dark circles", "dark spots", "fine lines and wrinkles",
                  "hydration", "normal skin", "oily skin", "pores", "redness", "sagging",
                  "sensitive skin", "uneven texture"]
    file_path = "all_attributes_modeling.csv"
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    # Preprocess data
    train_loader, val_loader = load_and_preprocess_data(file_path, attributes, tokenizer)

    # Initialize model, optimizer, and loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExplicitBTBERT(model_name="BAAI/bge-base-en-v1.5", num_labels=len(attributes)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-5)

    # Train classifier
    train_classifier(model, train_loader, optimizer, criterion, device, epochs=4)
    torch.save(model.state_dict(), "model.pth")
