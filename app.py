import streamlit as st
import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Connect to DuckDB
con = duckdb.connect('recipes.duckdb')

# Load data
df = con.execute("SELECT * FROM recipes").df()

# Vectorize ingredients
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ingredients'])

def recommend_recipes(user_ingredients):
    # Vectorize user input
    user_vector = vectorizer.transform([' '.join(user_ingredients)])
    # Compute cosine similarity
    similarities = cosine_similarity(user_vector, X)
    # Get top 5 similar recipes
    top_indices = similarities.argsort()[0, -5:][::-1]
    return df.iloc[top_indices]

# Streamlit app
st.title("Recipe Recommender")

# User input
user_ingredients = st.text_input("Enter ingredients (comma separated)").split(',')

if st.button("Recommend"):
    if user_ingredients:
        user_ingredients = [ingredient.strip() for ingredient in user_ingredients]
        recommendations = recommend_recipes(user_ingredients)
        st.write("Top recipe recommendations:")
        for idx, row in recommendations.iterrows():
            st.write(f"{row['title']} - {row['description']}")
    else:
        st.write("Please enter some ingredients.")



