import numpy as np

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

def euclidean_distance(embedding1, embedding2):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance

def manhattan_distance(embedding1, embedding2):
    distance = np.sum(np.abs(embedding1 - embedding2))
    return distance
