import numpy as np
import requests
import json
import csv
import pandas as pd
from bs4 import BeautifulSoup
import time
import util
import generate_similarity_PCA as PCA
import math
import constants

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial

def find_all_pro_similarity(pid_noob):
    with open('pp_list.json', 'r') as file:
        data = json.load(file)

    with open('pp_data_hero.json', 'r') as file:
        pp_hero_data = json.load(file)


    pp_hero_list = pp_hero_data
    pp_list = data["players"]

    similarity_arr = []

    noob_vector = get_noob_hero_list(pid_noob)

    for player in pp_list:
        player_id = player["id"]
        try:
            pro_list = pp_hero_list[str(player_id)]
        except:
            print("skipping", player_id)
            continue

        pro_vector = calculate_player_vector(pro_list)

        similarity_score = 1 - sim_algo2(noob_vector, pro_vector)
        print(player_id, ": ", similarity_score)
        similarity_arr.append({"id": player_id, "similarity_score": similarity_score})

    sorted_list = sorted(similarity_arr, key=lambda x: x['similarity_score'], reverse=True)

    mini_list = sorted_list[:5]
    sorted_mini_list = []
    result_dict = {item['id']: item['name'] for item in pp_list}
    for val in mini_list:
        pid = val['id']
        player_name = result_dict[pid]
        sorted_mini_list.append({pid: {"similarity_score": val['similarity_score'], "name": player_name}})

    return sorted_mini_list

    
# def sim_algo1(pid_1_list, pid_2_list):
#     calc = PCA.HeroSimilarityCalculatorPCA()
#     with open('pp_data_hero.json', 'r') as file:
#         pp_dict = json.load(file)
    
#     p1_vector = calculate_player_vector(pid_1_list)
#     p2_vector = calculate_player_vector(pid_2_list)

#     #this is the step that is most suspect
#     role_adjustment_matrix = calc.heroes_embeddings 
#     p1_role_adjusted = np.dot(role_adjustment_matrix, p1_vector)
#     p2_role_adjusted = np.dot(role_adjustment_matrix, p2_vector)


#     similarity_score = cosine_similarity([p1_role_adjusted], [p2_role_adjusted])[0, 0]

#     return similarity_score

def sim_algo2(pid_1_list, pid_2_list):
    
    return spatial.distance.cosine(pid_1_list, pid_2_list)

def get_noob_hero_list(pid_noob):
    hero_list = util.get_last_matches_hero_noob(pid_noob, constants.num_games)
    return calculate_player_vector(hero_list = hero_list)


def calculate_player_vector(hero_list):
    pp_vector = [0] * constants.hero_num


    with open('internal_hero_ids.json', 'r') as file:
        pp_dict = json.load(file)

    hero_ids = [hero['id'] for hero in pp_dict['data']['constants']['heroes']]
    print(hero_list)

    for val in hero_list:

        if val == 0:
            continue
        pp_vector[hero_ids.index(val)] += 1


    return pp_vector
