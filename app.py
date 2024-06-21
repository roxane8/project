### Building the Streamlit App
import streamlit as st
import duckdb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.title("Recipe Finder")
st.write("Enter the ingredients you have at home, and we'll show you possible recipes.")

user_ingredients = st.text_input("Enter ingredients separated by commas (e.g., tomato, cheese, pasta):")

@st.cache(hash_funcs={duckdb.DuckDBPyConnection: lambda _: None, KMeans: lambda _: None})
def load_and_cluster_data():
    con = duckdb.connect('recipes.duckdb')
    recipes_df = con.execute("SELECT * FROM recipes").fetchdf()

    # Combine all ingredients into a single string per recipe
    recipes_df['all_ingredients'] = recipes_df['ingredients'].apply(lambda x: ' '.join(eval(x)))

    # Vectorize the ingredients
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(recipes_df['all_ingredients'])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=42)
    recipes_df['cluster'] = kmeans.fit_predict(X)

    return recipes_df, vectorizer, kmeans

#recipes_df, vectorizer, kmeans = load_and_cluster_data()

def find_recipes(user_ingredients, recipes_df, vectorizer, kmeans):
    user_ingredients_list = [ingredient.strip().lower() for ingredient in user_ingredients.split(',')]
    user_ingredients_str = ' '.join(user_ingredients_list)

    user_vector = vectorizer.transform([user_ingredients_str])
    user_cluster = kmeans.predict(user_vector)[0]

    possible_recipes = recipes_df[recipes_df['cluster'] == user_cluster]

    return possible_recipes['title'].tolist()

if st.button("Find Recipes"):
    if user_ingredients:
        possible_recipes = find_recipes(user_ingredients, recipes_df, vectorizer, kmeans)
        if possible_recipes:
            st.write("Here are some recipes you can make:")
            for recipe in possible_recipes:
                st.write(f"- {recipe}")
        else:
            st.write("No recipes found with the given ingredients.")
    else:
        st.write("Please enter some ingredients.")