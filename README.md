# **Beauty Beyond Words: Multi-Label Ingredient-Attribute Classification**

This repository implements a multi-label classification pipeline to predict skincare product attributes (e.g., acne, hydration) based on textual metadata (ingredients and product titles). The system uses a BERT-based model, integrated explainability techniques (LIME, SHAP, Integrated Gradients), and robust evaluation metrics to provide interpretable insights into ingredient-attribute relationships.

---

## **Repository Structure**

### **1. Data Generation (`data_generation.py`)**
   - **Purpose**: Extract and preprocess data from the UC San Diego Amazon Product Reviews dataset to create a curated dataset for multi-label classification tasks.
   - **Key Features**:
     - Parses product metadata (titles, descriptions, and reviews).
     - Extracts ingredient lists using regular expressions and a predefined glossary of skincare terms.
     - Maps product attributes (e.g., acne, hydration) to binary labels using a dictionary-based entity recognition approach.
     - Generates a combined dataset of ingredients, product titles, and their associated attributes for modeling.
   - **Output**:
     - A formatted CSV file containing:
       - Input fields: Combined ingredients and product titles.
       - Target labels: Binary labels for attributes.
     - Ready-to-use dataset for the preprocessing and training scripts.

### **2. Preprocessing (`preprocessing.py`)**
   - **Purpose**: Prepare datasets for multi-label classification.
   - **Key Features**:
     - Data cleaning, tokenization, and combining product metadata.
     - Train-validation split creation.
     - Dataset class (`MultiLabelDataset`) to prepare tokenized data for modeling.
   - **Output**:
     - Train and validation datasets in PyTorch `DataLoader` format.

### **3. Training (`train.py`)**
   - **Purpose**: Train the classification model.
   - **Key Features**:
     - Model: BERT-based explicit architecture.
     - Training Phases:
       1. Freeze the BERT backbone and train the classifier head.
       2. Fine-tune the full model.
     - Binary cross-entropy loss for multi-label classification.
     - AdamW optimizer with dynamic learning rates.
   - **Output**:
     - Trained model weights.
     - Training and validation loss logs.

### **4. Evaluation (`eval.py`)**
   - **Purpose**: Evaluate the model and generate explainability insights.
   - **Key Features**:
     - Performance metrics:
       - Overall: F1 score, precision, recall, and accuracy.
       - Class-wise metrics.
     - Confusion matrices for each attribute.
     - Explainability tools:
       - LIME: Locally interpretable importance scores for ingredients.
       - SHAP: Global feature importance.
       - Integrated Gradients: Token-level attributions.
   - **Output**:
     - Metrics and class-wise confusion matrices.
     - Explainability insights for key ingredients and attributes.

### **5. Explainability (`explainability_metrics/`)**
   - **Purpose**: Scripts for generating explainability results.
   - **Key Tools**:
     - **LIME**: Local feature explanations.
     - **SHAP**: Shapley values for global insights.
     - **Integrated Gradients**: Gradient-based token attributions.

---

## **How to Use**

### **1. Download product metadata and reviews from UC San Diego repository and save to accessible directory

### **2. Data Generation**
  - Run the `data_generation.py` script:
    ```bash
     python data_generation.py
     ```
    
### **2. Preprocessing**
   - Run the `preprocessing.py` script:
     ```bash
     python data_generation.py
     ```

### **3. Training**
   - Train the model:
     ```bash
     python train.py
     ```

### **4. Evaluation**
   - Evaluate the trained model:
     ```bash
     python eval.py
     ```

### **5. Run explainability scripts

---

## **Requirements**

- Python 3.8 or later
- Libraries:
  - `torch`, `transformers`, `scikit-learn`, `pandas`, `tqdm`, `lime`, `matplotlib`
- GPU (optional but recommended for training).

---

## **Explainability**

- **LIME**:
  - Highlights local ingredient importance for each product.
- **SHAP**:
  - Provides global insights into ingredient contributions.
- **Integrated Gradients**:
  - Analyzes token-level contributions using gradients.

These techniques enhance transparency and allow practitioners to trace attribute predictions back to specific ingredients.

---

## **Future Work**

- **Enhanced Data Quality**:
  - Use domain-specific labels or collaborate with industry experts.
- **Larger Datasets**:
  - Train with more reviews per product for better class representation.
- **Advanced Architectures**:
  - Experiment with domain-adapted transformers and semi-supervised techniques.

---

## **Contact**

For questions or feedback, contact:

- **Shreya Sriram**: ss3589@cornell.edu
- **Zachary Decker**: zad25@cornell.edu
- **Priyanshi Gupta**: pg485@cornell.edu

---

## **License**

This project is licensed under the MIT License.
