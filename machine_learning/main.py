from generate_similarity_Embedding import HeroSimilarityCalculatorEMB
from generate_similarity_PCA import HeroSimilarityCalculatorPCA
import util as util
def main():

    match_int = 7531916495
    match_list = [7533640643, 7531916495]
    # print(util.check_real_game(match_int))
    print(util.check_real_game(match_list))

    # util.generate_training_data()
    # EMB_calculator = HeroSimilarityCalculatorEMB()
    # PCA_calculator = HeroSimilarityCalculatorPCA()

    # hero1_id = 41
    # hero2_id = 44

    # similarity_measure = 'cosine'
    # EMB_similarity = EMB_calculator.calculate_similarity(hero1_id, hero2_id,  'cosine')
    # PCA_similarity = PCA_calculator.calculate_similarity(hero1_id, hero2_id,  'cosine')

    # print(f"EMB Similarity: {hero1_id} and {hero2_id} using {similarity_measure} similarity: {EMB_similarity}")
    # print(f"PCA Similarity: {hero1_id} and {hero2_id} using {similarity_measure} similarity: {PCA_similarity}")

if __name__ == "__main__":
    main()
