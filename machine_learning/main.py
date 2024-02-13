# from generate_similarity_Embedding import HeroSimilarityCalculatorEMB
from generate_similarity_PCA import HeroSimilarityCalculatorPCA
from ml_util import cosine_similarity
import util as util
import ml
import constants
import json
import pandas as pd


#1131498310 - Blake
def main():

    print("Starting Program")
    print(ml.find_all_pro_similarity(1131498310))

    
    # h = HeroSimilarityCalculatorPCA()    

    # with open('internal_hero_ids.json', 'r') as file:
    #     pp_dict = json.load(file)

    # hero_ids = [hero['id'] for hero in pp_dict['data']['constants']['heroes']]

    # print(hero_ids)

    # # Creating an empty DataFrame
    # df = pd.DataFrame(columns=["Heroes", "Similarity"])

    # for i in range(len(hero_ids)):
    #     for j in range(i + 1, len(hero_ids)):  # Start from i+1 to avoid duplicates and similarity with itself
    #         similarity = h.calculate_similarity(hero_ids[i], hero_ids[j], 'cosine')

    #         # Concatenating data to the DataFrame
    #         df = pd.concat([df, pd.DataFrame({"Heroes": [[hero_ids[i], hero_ids[j]], [hero_ids[j], hero_ids[i]]], "Similarity": [similarity, similarity]})], ignore_index=True)

    # # Sorting DataFrame by "Similarity" column in descending order
    # df = df.sort_values(by="Similarity", ascending=False)

    # # Drop duplicate rows
    # df = df.drop_duplicates(subset=["Heroes"])

    # # Save DataFrame to CSV
    # df.to_csv('similarity.csv', index=False)

if __name__ == "__main__":
    main()
