import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CUT_OFF_SCORE = 0.0

# Load data function with caching to optimize performance
@st.cache_data  # Updated cache function for Streamlit's latest version
def load_data():
    """
    Load the processed dataset from the specified path.
    """
    return pd.read_csv('../data/Processed_df.csv')

# Function to filter recipes based on user preferences
def filter_recipes(df, donts, min_rating, max_time, min_servings, max_servings, include_meat, include_dairy):
    """
    Apply filtering on the dataset based on user preferences.
    """
    meat_ingredients = ['chicken', 'pork', 'lamb', 'goat', 'beef', 'bacon']
    dairy_ingredients = ['milk', 'cheese', 'butter', 'cream', 'yogurt', 'egg']
    
    # Ensure 'donts' is always a list
    if not donts:
        donts = []
    
    # Convert 'donts' to lowercase for case-insensitive matching
    donts = [dont.lower() for dont in donts]
    
    # Add meat or dairy ingredients to 'donts' if excluded
    if not include_meat:
        donts.extend(meat_ingredients)
    if not include_dairy:
        donts.extend(dairy_ingredients)
    
    # Filter out recipes that contain any ingredients in 'donts'
    df_filtered = df[~df['processed_ingredients'].apply(lambda x: any(dont in x.lower() for dont in donts))]
    
    # Apply rating filter
    df_filtered = df_filtered[df_filtered['rating'] >= min_rating]
    
    # Apply maximum time filter, handling missing values
    if max_time is not None:
        df_filtered = df_filtered[(df_filtered['total_time'] == 'missing') | (df_filtered['total_time'].apply(lambda x: int(x) if x.isdigit() else np.inf) <= max_time)]
    
    # Apply servings filter
    if min_servings is not None:
        df_filtered = df_filtered[df_filtered['servings'] >= min_servings]
    if max_servings is not None:
        df_filtered = df_filtered[df_filtered['servings'] <= max_servings]
    
    return df_filtered

# Function to generate recipe recommendations
def recommend_recipes(df_filtered, query):
    """
    Compute cosine similarity between user query and recipes to find the best matches.
    """
    if df_filtered.empty:
        return pd.DataFrame()
    
    # Initialize and fit the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    tfidf_matrix_filtered = tfidf_vectorizer.fit_transform(df_filtered['processed_ingredients'])
    
    # Transform the user query
    query_vector = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarities
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix_filtered).flatten()
    
    # Get the top 5 most similar recipes with a cutoff similarity score
    top_indices = [i for i in cosine_similarities.argsort()[-5:][::-1] if cosine_similarities[i] >= CUT_OFF_SCORE]
    recommendations = df_filtered.iloc[top_indices][['recipe_name', 'rating', 'total_time', 'servings', 'ingredients', 'directions', 'nutrition', 'url']]
    recommendations['similarity_score'] = cosine_similarities[top_indices]
    
    return recommendations

# Streamlit UI setup
def main():
    """
    Streamlit app main function that handles user interaction.
    """
    st.title('Recipe Recommendation System')
    
    df = load_data()
    
    # User inputs
    user_query = st.text_input("What ingredients you want to have in your recipe?", "apple")
    user_donts = st.text_input("Ingredients to avoid (comma-separated)", "")
    min_rating = st.slider("Minimum rating", 0.0, 5.0, 4.0, 0.1)
    max_time = st.slider("Maximum cooking time (minutes)", 0, 180, 60)
    min_servings = st.slider("Minimum servings", 1, 10, 2)
    max_servings = st.slider("Maximum servings", 1, 30, 6)
    include_meat = st.radio("Is it OK to recommend with meat?", ("Yes", "No")) == "Yes"
    include_dairy = st.radio("Diary preferance?", ("Yes", "No")) == "Yes"
    
    if st.button('Recommend Recipes'):
        # Process user inputs
        donts_list = [x.strip() for x in user_donts.split(',')] if user_donts.strip() else []
        df_filtered = filter_recipes(df, donts_list, min_rating, max_time, min_servings, max_servings, include_meat, include_dairy)
        recommendations = recommend_recipes(df_filtered, user_query)
        
        if recommendations.empty:
            st.write("No recipes found.")
        else:
            st.write(recommendations)

if __name__ == "__main__":
    main()
