import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from similarity_measures import cosine_similarity, euclidean_distance, manhattan_distance, minkowski_distance

# Read data from CSV file
csv_file_path = 'hero_data.csv'
df = pd.read_csv(csv_file_path)

# Extract numerical columns for PCA
numerical_columns = df.columns[1:]  # First column is heroId
data_for_pca = df[numerical_columns].values

# Perform PCA
pca = PCA(n_components=10)
heroes_embeddings = pca.fit_transform(data_for_pca)

# Function to get the embedding for a hero
def get_hero_embedding(hero_id):
    index = df.index[df['heroId'] == hero_id].tolist()[0]
    return heroes_embeddings[index]

# Function to calculate similarity between two heroes
def calculate_similarity(hero1_id, hero2_id, similarity_measure):
    embedding1 = get_hero_embedding(hero1_id)
    embedding2 = get_hero_embedding(hero2_id)

    if similarity_measure == 'cosine':
        similarity = cosine_similarity(embedding1, embedding2)
    elif similarity_measure == 'euclidean':
        similarity = euclidean_distance(embedding1, embedding2)
    elif similarity_measure == 'manhattan':
        similarity = manhattan_distance(embedding1, embedding2)

    else:
        raise ValueError(f"Invalid similarity measure: {similarity_measure}")

    return similarity

# Example usage
hero1_id = 104
hero2_id = 2
similarity_measure = 'cosine'  # Change this to the desired similarity measure
similarity = calculate_similarity(hero1_id, hero2_id, similarity_measure)

print(f"Similarity between Hero {hero1_id} and Hero {hero2_id} using {similarity_measure} similarity: {similarity}")
