import json
import pandas as pd
from spacy.matcher import PhraseMatcher
import spacy
import re
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

tqdm.pandas()

data = []
with open('test.jsonl', 'r') as f:
    for line in f:
        try:
            # Parse each line as a JSON object
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping line due to error: {e}")

reviews = pd.DataFrame.from_records(data)

data = []
with open('test_meta.jsonl', 'r') as f:
    for line in f:
        try:
            # Parse each line as a JSON object
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping line due to error: {e}")


products = pd.DataFrame.from_records(data)

products = products[(products['description'].apply(lambda x: len(x)  > 0)) | (products['features'].apply(lambda x: len(x)  > 0))]

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


ingredients_pattern = "|".join([re.escape(ingredient.lower()) for ingredient in ingredients])
ingredients_regex = re.compile(ingredients_pattern)

def extract_unique_ingredients_regex(description, ingredients_regex):
    """
    Extract unique ingredients dynamically from a description using regex.
    """
    matches = ingredients_regex.findall(description.lower())
    return sorted(set(matches))

# Combine 'features' and 'description' columns
products['combined_description'] = products['features'].astype(str) + " " + products['description'].astype(str)

# Extract ingredients using regex with progress bar
products['extracted_ingredients'] = products['combined_description'].progress_apply(
    lambda desc: extract_unique_ingredients_regex(desc, ingredients_regex)
)
products = products[products['extracted_ingredients'].apply(lambda x: len(x) > 0)]

products = products.drop_duplicates(subset=['description'])
reviews_match = products.merge(reviews, on='parent_asin')

# Generalized skin features with related phrases
skin_features = {
    "Normal Skin": ["normal", "balanced", "healthy", "clear", "untroubled", "even"],
    "Oily Skin": ["oily", "greasy", "shiny", "excess sebum", "slick", "glossy"],
    "Combination Skin": ["combination", "mixed", "dry and oily", "dual type", "patchy oily"],
    "Sensitive Skin": ["sensitive", "irritation", "reactive", "allergic", "fragile", "delicate", "easily irritated", "prone to redness"],
    "Acne": ["acne", "pimple", "blemish", "breakout", "zits", "cystic acne", "spots", "acne-prone", "comedones"],
    "Hydration": ["hydrating", "moisture", "moisturizing", "replenish", "quenched", "hydrated", "plumping", "water retention", "moist"],
    "Pores": ["pores", "enlarged pores", "pore size", "clogged pores", "minimize pores", "visible pores", "pore congestion"],
    "Fine Lines and Wrinkles": ["wrinkles", "fine lines", "aging", "anti-aging", "crow's feet", "expression lines", "laugh lines", "age lines", "crinkles"],
    "Sagging": ["sagging", "loose", "loss of firmness", "drooping", "lack of elasticity", "laxity", "lifting", "skin slackening", "gravity-prone"],
    "Dark Spots": ["dark spots", "hyperpigmentation", "discoloration", "sun spots", "age spots", "melasma", "uneven tone", "brown patches", "pigmented spots"],
    "Redness": ["redness", "red patches", "inflammation", "rosacea", "flushed", "red blotches", "irritated skin", "blushing", "hyperemia"],
    "Uneven Texture": ["uneven texture", "rough", "bumpy", "textured", "dull", "grainy", "coarse", "patchy", "irregular texture"],
    "Dark Circles": ["dark circles", "under-eye", "eye bags", "puffy eyes", "tired eyes", "under-eye discoloration", "shadow", "hollows", "dark under-eyes"]
}

# Load spaCy model with pipeline components disabled for efficiency
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])

# Function to extract unique skin features using exact and related matching
def extract_unique_skin_features(text, nlp, skin_features):
    """
    Extract unique skin features dynamically from text using efficient matching.

    Args:
    - text (str): The input text to extract skin features from.
    - nlp (spacy.Language): The spaCy NLP pipeline.
    - skin_features (dict): Dictionary of skin features and related phrases.

    Returns:
    - list: List of unique skin features (case-insensitive).
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # Case-insensitive matching

    # Add patterns for exact and related matches
    for feature, related_phrases in skin_features.items():
        patterns = [nlp.make_doc(phrase) for phrase in [feature] + related_phrases]
        matcher.add(feature, patterns)

    # Process the text
    doc = nlp(text)
    matches = matcher(doc)

    # Extract matched skin features using exact and related phrases
    unique_features = {nlp.vocab.strings[match_id].lower() for match_id, _, _ in matches}

    return sorted(unique_features)


reviews_match['skin_features'] = reviews_match['text'].progress_apply(
    lambda review: extract_unique_skin_features(str(review), nlp, skin_features)
)

reviews_match = reviews_match[reviews_match['skin_features'].apply(lambda x: len(x) > 0)]

reviews_match['extracted_ingredients'] = reviews_match['extracted_ingredients'].apply(tuple)
grouped_reviews = reviews_match.groupby(['parent_asin', 'extracted_ingredients'])['skin_features'].apply(lambda x: [attr for sublist in x for attr in sublist])
grouped_reviews = grouped_reviews.apply(lambda x: list(set(x)))
grouped_reviews_df = grouped_reviews.reset_index()


# One-hot encode extracted ingredients
ingredients = [set(ing) for ing in grouped_reviews_df['extracted_ingredients']]
mlb_ingredients = MultiLabelBinarizer()
ingredients_encoded = mlb_ingredients.fit_transform(ingredients)
ingredients_df = pd.DataFrame(ingredients_encoded, columns=mlb_ingredients.classes_)

# Multi-label encode skin features
skin_features = [set(features) for features in grouped_reviews_df['skin_features']]
mlb_features = MultiLabelBinarizer()
features_encoded = mlb_features.fit_transform(skin_features)
features_df = pd.DataFrame(features_encoded, columns=mlb_features.classes_)

# Combine into a single DataFrame
modeling_df = pd.concat([grouped_reviews_df['parent_asin'], ingredients_df, features_df], axis=1)

print(modeling_df)