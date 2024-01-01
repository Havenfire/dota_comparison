from generate_similarity_Embedding import HeroSimilarityCalculatorEMB
from generate_similarity_PCA import HeroSimilarityCalculatorPCA

def main():
    hero_sim_calculator = HeroSimilarityCalculatorPCA()

    hero1_id = 16
    hero2_id = 10
    similarity_measure = 'cosine'
    similarity = hero_sim_calculator.calculate_similarity(hero1_id, hero2_id, similarity_measure)

    print(f"Similarity between Hero {hero1_id} and Hero {hero2_id} using {similarity_measure} similarity: {similarity}")

if __name__ == "__main__":
    main()
