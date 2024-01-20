import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from ml_util import cosine_similarity, euclidean_distance, manhattan_distance

class HeroSimilarityCalculatorPCA:
    def __init__(self, csv_file_path='hero_data.csv', n_components=10):
        self.df = pd.read_csv(csv_file_path)

        # Extract numerical columns for PCA
        numerical_columns = self.df.columns[1:]  # First column is heroId
        data_for_pca = self.df[numerical_columns].values

        # Perform PCA
        self.pca = PCA(n_components=n_components)
        self.heroes_embeddings = self.pca.fit_transform(data_for_pca)

        # Save embeddings to a file
        print("Saving PCA_embeddings")
        np.save('embeddings_PCA.npy', self.heroes_embeddings)

    def get_hero_embedding(self, hero_id):
        index = self.df.index[self.df['heroId'] == hero_id].tolist()[0]
        return self.heroes_embeddings[index]

    def calculate_similarity(self, hero1_id, hero2_id, similarity_measure):
        embedding1 = self.get_hero_embedding(hero1_id)
        embedding2 = self.get_hero_embedding(hero2_id)

        if similarity_measure == 'cosine':
            similarity = cosine_similarity(embedding1, embedding2)
        elif similarity_measure == 'euclidean':
            similarity = euclidean_distance(embedding1, embedding2)
        elif similarity_measure == 'manhattan':
            similarity = manhattan_distance(embedding1, embedding2)
        else:
            raise ValueError(f"Invalid similarity measure: {similarity_measure}")

        return similarity
