#%%
import pandas as pd
import numpy as np
#Load the data
df=pd.read_csv("../data/recipes.csv")
#removing duplicates =df1
df1 = df.drop_duplicates(subset='recipe_name', keep='first')
df1.to_csv("../data/recipes_cleaned.csv", index=False)

print("Duplicates removed. The DataFrame now has", len(df1), "rows.")
#dropping columns Url and cusine path
df1 = df1.drop(columns=['Unnamed: 0', 'img_src', 'cuisine_path', 'prep_time', 'cook_time'])
#%%
#checking for null values
nan_counts = df1.isna().sum()
print(nan_counts)

#%%
#Handling missing values in time and yield
#function to convert time from hrs to minutes
import re

def convert_to_minutes(time_str):
    if pd.isna(time_str):
        return 'missing'  # Return 'missing' for NaN values

    # Regular expressions to find hours and minutes
    hours = re.search(r'(\d+)\s*hr', time_str)
    minutes = re.search(r'(\d+)\s*min', time_str)
    
    total_minutes = 0
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))
    
    return total_minutes

df1['total_time'] = df1['total_time'].apply(convert_to_minutes)
df1['yield'] = df1['yield'].fillna('missing')
# %%
print("The DataFrame now has", len(df1), "rows.")
# %%
df1.head()
# %%
# combining both ingredients and directions together on one columns and removing the duplicates for avoiding future errors
def combine_and_remove_duplicates(ingredients, directions):
    # Combine both strings and split into words
    combined = f"{ingredients} {directions}".lower()
    words = combined.split()
    
    # Remove duplicates by converting the list to a set and back to list
    unique_words = list(set(words))
    
    # Join the unique words back into a single string
    return ' '.join(unique_words)

# Apply the function to create a new column
df1['combined_ingredients_directions'] = df1.apply(lambda x: combine_and_remove_duplicates(x['ingredients'], x['directions']), axis=1)

# %%
# testing how our feature looks
row_index = 0  # Index of the row you want to inspect
row_data = df1.loc[row_index, ['ingredients', 'directions', 'combined_ingredients_directions']]
print("Ingredients:\n", row_data['ingredients'])
print("\nDirections:\n", row_data['directions'])
print("\nCombined Ingredients and Directions:\n", row_data['combined_ingredients_directions'])


#%% Preprocessing the text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
#%%
# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# Load SpaCy's English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_sm')

# Initialize NLTK's Porter Stemmer
stemmer = PorterStemmer()

# Ensure contractions is installed
# pip install contractions
nlp = spacy.load('en_core_web_sm')

def preprocess_text_spacy(text):
    # Process the text with SpaCy
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove numbers and specific punctuations like parentheses
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation except for underscores and spaces

    doc = nlp(text)
    
    # List comprehension for processing tokens
    tokens = [
        token.lemma_.lower()  # Apply lowercasing to the lemmatized token
        for token in doc
        if not token.is_stop and not token.is_punct and token.pos_ != 'PRON'
    ]
    
    # Join the tokens back into a single string
    return ' '.join(tokens)
# %%
# Apply preprocessing
df1['processed_ingredients_directions'] = df1['combined_ingredients_directions'].apply(preprocess_text_spacy)

# %%
df1.drop('combined_ingredients_directions', axis=1, inplace=True)

# %%
df1['processed_ingredients'] = df1['ingredients'].apply(preprocess_text_spacy)
def remove_duplicate_words(text):
    words = text.split()
    seen = set()
    seen_add = seen.add
    # Add to seen and keep only unseen words to preserve order
    unique_words = [x for x in words if not (x in seen or seen_add(x))]
    return ' '.join(unique_words)
df1['processed_ingredients'] = df1['processed_ingredients'].apply(remove_duplicate_words)
# %%
df1.head()
# %%
df1.to_csv('../data/Processed_df.csv', index=False)
