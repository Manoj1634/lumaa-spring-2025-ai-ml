#%%
#base_recommendation
import pandas as pd
import numpy as np
df=pd.read_csv('/home/ubuntu/pro1/data/Processed_df.csv')
# %%
df.head()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# %%
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['ingredients'])
# %%
def recommend_recipes(query):
    # Transform the query to the same vector space as the ingredients
    query_vector = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarity between query and all recipes
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the top 5 most similar recipes
    top_indices = cosine_similarities.argsort()[-5:][::-1]  # Indices of recipes sorted by similarity
    recommendations = df.loc[top_indices, ['recipe_name', 'ingredients']]
    recommendations['similarity_score'] = cosine_similarities[top_indices]
    
    return recommendations

# %%
user_query = "I dont want to have something with apples"
recommendations = recommend_recipes(user_query)
print(recommendations)

# %%
