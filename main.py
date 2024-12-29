import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv("recipes_Dataset.csv", low_memory=False)
df = df.dropna()

# Combine text features into a single column
df['combined_features'] = (
    df['tags'] + ' ' + df['ingredients'] + ' ' + df['diet_restrictions'].fillna('')
)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Compute cosine similarity between recipes
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on user inputs
def get_recommendations(user_inputs, cosine_sim=cosine_sim, df=df):
    # Filter the dataframe based on user inputs
    filtered_df = df.copy()
    for feature, values in user_inputs.items():
        if feature == 'ingredients':
            filtered_df = filtered_df[
                filtered_df['ingredients'].apply(
                    lambda x: any(ingredient.strip() in x for ingredient in values)
                )
            ]
        elif feature == 'min_rating':
            filtered_df = filtered_df[filtered_df['rating'] >= values]
        else:
            filtered_df = filtered_df[
                filtered_df[feature].apply(
                    lambda x: any(val.strip() in x for val in values)
                )
            ]
    
    # Get indices of filtered recipes
    indices = filtered_df.index
    
    # Initialize a list to store recommendations
    recommendations = []
    
    for idx in indices:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        most_similar_recipe_id = sim_scores[1][0] if len(sim_scores) > 1 else None
        recipe_details = (
            df.iloc[most_similar_recipe_id] if most_similar_recipe_id is not None else None
        )
        recommendations.append({
            'recipe_id': idx,
            'similar_recipe_id': most_similar_recipe_id,
            'recipe_details': recipe_details
        })
    
    return recommendations

# Streamlit App
st.title('Recipe Recommender')

# User Inputs
st.sidebar.title('User Inputs')
tags_input = st.sidebar.text_input('Tags (comma-separated)', 'easy, quick')
ingredients_input = st.sidebar.text_input('Ingredients (comma-separated)', 'chicken')
diet_restrictions_input = st.sidebar.text_input('Diet Restrictions (comma-separated)', 'gluten-free')
min_rating_input = st.sidebar.slider('Minimum Rating', min_value=0.0, max_value=5.0, step=0.1, value=4.5)

# Process user inputs
user_inputs = {
    'tags': tags_input.split(','),
    'ingredients': ingredients_input.split(','),
    'diet_restrictions': diet_restrictions_input.split(','),
    'min_rating': min_rating_input
}

# Get recommendations
recommendations = get_recommendations(user_inputs)

# Display recommendations
st.header('Recommendations')
if recommendations:
    for rec in recommendations:
        st.write(f"Recipe ID: {rec['recipe_id']}")
        if rec['recipe_details'] is not None:
            st.write("Similar Recipe Details:")
            st.write(rec['recipe_details'])
        else:
            st.write("No similar recipe found.")
else:
    st.write("No recommendations found based on your inputs.")
