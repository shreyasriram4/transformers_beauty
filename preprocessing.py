import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class MultiLabelDataset(Dataset):
    """
    Custom dataset for multi-label classification tasks.
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels
        }

def load_and_preprocess_data(file_path, attributes, tokenizer, batch_size=8):
    """
    Load and preprocess the dataset, create dataloaders for training and validation.

    Args:
        file_path: Path to the dataset CSV file.
        attributes: List of attributes to classify.
        tokenizer: Tokenizer instance.
        batch_size: Batch size for the DataLoader.

    Returns:
        train_loader, val_loader: DataLoaders for training and validation.
    """
    # Load dataset
    df = pd.read_csv(file_path)
    df = df.drop(['Unnamed: 0'], axis=1)

    # Combine text fields
    df['title'] = df['title'].fillna("")
    df['combined_text'] = df['Ingredients_Text'] + " [SEP] " + df['title']

    # Train-test split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['combined_text'].tolist(),
        df[attributes].values,
        test_size=0.2,
        random_state=42
    )

    # Create datasets
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
