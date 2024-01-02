from generate_similarity_Embedding import HeroSimilarityCalculatorEMB
from generate_similarity_PCA import HeroSimilarityCalculatorPCA

def main():
    EMB_calculator = HeroSimilarityCalculatorEMB()
    PCA_calculator = HeroSimilarityCalculatorPCA()

    hero1_id = 41
    hero2_id = 44

    similarity_measure = 'cosine'
    EMB_similarity = EMB_calculator.calculate_similarity(hero1_id, hero2_id,  'cosine')
    PCA_similarity = PCA_calculator.calculate_similarity(hero1_id, hero2_id,  'cosine')

    print(f"EMB Similarity: {hero1_id} and {hero2_id} using {similarity_measure} similarity: {EMB_similarity}")
    print(f"PCA Similarity: {hero1_id} and {hero2_id} using {similarity_measure} similarity: {PCA_similarity}")

if __name__ == "__main__":
    main()
