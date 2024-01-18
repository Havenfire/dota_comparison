from generate_similarity_Embedding import HeroSimilarityCalculatorEMB
from generate_similarity_PCA import HeroSimilarityCalculatorPCA
import util as util
def main():

    print("Starting Program")
    # print(util.get_popular_players())

    util.popular_players_past_games(num_games=100, num_players = 200)
    # print(util.get_last_games_hero(1131498310, 5))

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
