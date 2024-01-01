import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class HeroSimilarityCalculatorEMB:
    def __init__(self, csv_file_path='hero_data.csv', embedding_dim=10):
        self.df = pd.read_csv(csv_file_path)

        # Extract numerical columns
        numerical_columns = self.df.columns[1:]  # First column is heroId
        data = self.df[numerical_columns].values

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        # Split the data for training and validation
        train_data, val_data = train_test_split(scaled_data, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors
        train_data_tensor = torch.FloatTensor(train_data)
        val_data_tensor = torch.FloatTensor(val_data)

        # Build and train the autoencoder
        input_dim = data.shape[1]
        model = Autoencoder(input_dim, embedding_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(train_data_tensor)
            loss = criterion(output, train_data_tensor)
            loss.backward()
            optimizer.step()

        # Extract the embeddings
        self.heroes_embeddings = model.encoder(train_data_tensor).detach().numpy()

    def get_hero_embedding(self, hero_id):
        index = self.df.index[self.df['heroId'] == hero_id].tolist()[0]
        return self.heroes_embeddings[index]

    def calculate_similarity(self, hero1_id, hero2_id, similarity_measure):
        embedding1 = self.get_hero_embedding(hero1_id)
        embedding2 = self.get_hero_embedding(hero2_id)

        if similarity_measure == 'cosine':
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        elif similarity_measure == 'euclidean':
            similarity = euclidean_distances([embedding1], [embedding2])[0][0]
        elif similarity_measure == 'manhattan':
            similarity = manhattan_distances([embedding1], [embedding2])[0][0]
        else:
            raise ValueError(f"Invalid similarity measure: {similarity_measure}")

        return similarity

