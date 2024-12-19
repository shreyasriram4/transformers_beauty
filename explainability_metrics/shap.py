ingredients = [
    "Allantoin", "Alcohol Denat", "Almond Oil", "Aloe Vera", "Alpha Hydroxy Acid", "Amala Oil", "Amino Acids",
    "Amoxicillin", "Antioxidants", "Apple Cider Vinegar", "Apricot Kernel Oil", "Arbutin", "Argan Oil",
    "Argireline", "Ascorbyl Glucoside", "Astaxanthin", "Avocado Oil", "Azelaic Acid", "Azulene", "Baobab",
    "Baking Soda", "Bakuchiol", "Bentonite Clay", "Benzoyl Peroxide", "Benzyl Alcohol", "Beta Glucan",
    "Bhringraj Oil", "Biotin", "Bio Oil", "Black Cumin Seed Oil", "Borage Seed Oil", "Butylene Glycol", "CBD Oil",
    "CBD", "Caffeine", "Calamine Lotion", "Camellia Extract", "Capric Triglyceride", "Caprylyl Glycol", "Carbomer",
    "Caviar Extract", "Carrier Oils", "Carrot", "Castor Oil", "Cephalexin", "Ceramides", "Cetearyl Alcohol",
    "Chamomile", "Charcoal", "Chebula", "Chia Seed Oil", "Citric Acid", "Cocamidopropyl-Betaine", "Cocoa Butter",
    "Coconut Oil", "Collagen", "Colloidal Oatmeal", "Cone Snail Venom", "Copper Peptides", "CoQ10",
    "Cyclopentasiloxane", "Cypress Oil", "Desitin", "Dihydroxyacetone", "Dimethicone", "Doxycycline", "Emollients",
    "Emu Oil", "Epsom Salt", "Eucalyptus Oil", "Evening Primrose Oil", "Ferulic Acid", "Fermented Oils",
    "Frangipani", "Gluconolactone", "Glycerin", "Glycolic Acid", "Goat's Milk", "Goji Berry", "Gold",
    "Grapeseed Oil", "Green Tea", "Hemp Oil", "Homosalate", "Honey", "Humectants", "Hyaluronic Acid",
    "Hydrocortisone", "Hydrogen Peroxide", "Hydroquinone", "Isododecane", "Isoparaffin", "Isopropyl Myristate",
    "Jojoba", "Kaolin", "Karanja Oil", "Kigelia Africana", "Kojic Acid", "Kukui Nut Oil", "Lactic Acid",
    "Lactobionic Acid", "Lanolin", "Lavender Oil", "Lemon Juice", "Licorice Extract", "Lysine", "Madecassoside",
    "Magnesium", "Magnesium Aluminum Silicate", "Malic Acid", "Mandelic Acid", "Manuka Honey",
    "Marshmallow Root Extract", "Marula Oil", "Meadowfoam", "Methylparaben", "Mineral Oil", "Moringa Oil",
    "Murumuru Butter", "Muslin", "Neem Oil", "Niacinamide", "Nizoral", "Oat", "Octinoxate", "Octisalate",
    "Octocrylene", "Olive Oil", "Omega Fatty Acids", "Oxybenzone", "Panthenol", "Parabens", "Peppermint Oil",
    "Petroleum Jelly", "PHA", "Phenoxyethanol", "Phytic Acid", "Phytosphingosine", "Placenta", "Plum Oil",
    "Polyglutamic Acid", "Polypeptides", "Pomegranates", "Prickly Pear Oil", "Probioitics", "Progeline",
    "Propanediol", "Propolis", "Propylene Glycol", "Propylparabens", "Purslane Extract", "Pycnogenol",
    "Quercetin", "Reishi Mushrooms", "Resveratrol", "Retin-A", "Retinaldehyde", "Retinol", "Retinyl Palmitate",
    "Rosehip Oil", "Rosemary", "Royal Jelly", "Safflower Oil", "Salicylic Acid", "Sea Buckthorn Oil", "Sea Salt",
    "Seaweed", "Sea Moss", "Shea Butter", "Silver", "Snail Mucin", "Sodium Ascorbyl Phosphate", "Sodium Deoxycholate",
    "Sodium Hyaluronate", "Sodium Hydroxide", "Sodium Lauroyl Lactylate", "Sodium Lauryl Sulfate", "Sodium Palmate",
    "Sodium PCA", "Sodium Tallowate", "Soybean Oil", "Spironolactone", "Stearic Acid", "Stearyl Alcohol",
    "Squalane", "Stem Cells", "Succinic Acid", "Sulfates", "Sulfur", "Sunflower Oil", "Synthetic Beeswax", "Talc",
    "Tamanu Oil", "Tea Tree Oil", "Tepezcohuite", "Tranexamic Acid", "Tretinoin", "Triethanolamine", "Turmeric",
    "Undecylenic Acid", "Urea 40", "Vegetable Glycerin", "Vitamin A", "Vitamin B3", "Vitamin C", "Vitamin D",
    "Vitamin E", "Vitamin F", "Vitamin K", "Volcanic Ash", "Willow Bark Extract", "Xanthan Gum", "Zinc"
]

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Define attributes and load model/tokenizer
attributes = ["acne", "combination skin", "dark circles", "dark spots", "fine lines and wrinkles", 
              "hydration", "normal skin", "oily skin", "pores", "redness", "sagging", 
              "sensitive skin", "uneven texture"]
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-base-en-v1.5', num_labels=13)
model.load_state_dict(torch.load("BAAI_bge-base-en-v1.5_model_weights.pth"))
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the LIME prediction function
def predict_function(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs

# Create an instance of the LIME text explainer
explainer = LimeTextExplainer(class_names=attributes)

# Function to chunk the list
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Process ingredients in chunks of 20
# Dictionary to store top ingredients for each attribute across all chunks
all_top_ingredients = {attribute: [] for attribute in attributes}

for i, chunk in enumerate(chunk_list(ingredients, 20)):
    sample_text = ", ".join(chunk)
    
    try:
        # Get explanation for this chunk
        explanation = explainer.explain_instance(
            sample_text, 
            predict_function, 
            num_features=len(chunk),
            num_samples=500,
            labels=list(range(len(attributes)))  # Explicitly specify all label indices
        )
        
        # Save explanation as HTML
        with open(f'lime_explanation_chunk_{i+1}.html', 'w', encoding='utf-8') as f:
            f.write(explanation.as_html())
        
        # Process results for this chunk
        for label_idx, attribute in enumerate(attributes):
            try:
                feature_importances = explanation.as_list(label=label_idx)
                # Filter to only include actual ingredients from this chunk
                relevant_importances = [
                    (feat, score) for feat, score in feature_importances 
                    if any(ingredient.lower() in feat.lower() for ingredient in chunk)
                ]
                # Add to running list for this attribute
                all_top_ingredients[attribute].extend(relevant_importances)
            except KeyError as e:
                print(f"Could not get explanations for {attribute} (label {label_idx})")
                continue
    
    except Exception as e:
        print(f"Error processing chunk {i+1}: {str(e)}")
        continue
import matplotlib.pyplot as plt

# ... existing code ...

# Function to plot top ingredients for each attribute
def plot_top_ingredients(attribute, top_ingredients):
    ingredients, scores = zip(*top_ingredients)
    plt.figure(figsize=(10, 6))
    plt.barh(ingredients, scores, color='skyblue')
    plt.xlabel('Importance Score')
    plt.title(f'Top 10 Ingredients for {attribute}')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest score on top
    plt.tight_layout()
    plt.show()


# Print top 4 ingredients across all chunks for each attribute
# Print and plot top 10 ingredients across all chunks for each attribute
print("\nTop 10 ingredients across all chunks:")
for attribute in attributes:
    print(f"\n{attribute}:")
    # Sort all ingredients for this attribute
    sorted_ingredients = sorted(all_top_ingredients[attribute], key=lambda x: x[1], reverse=True)
    
    print("Positive correlations:")
    positive = [x for x in sorted_ingredients if x[1] > 0][:10]  # Change to top 10
    for ingredient, score in positive:
        print(f"{ingredient}: {score:.4f}")
    
    # Plot the top 10 positive correlations
    plot_top_ingredients(attribute, positive)
