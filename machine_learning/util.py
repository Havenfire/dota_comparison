import numpy as np
import requests
import json
import csv
import pandas as pd



API_URL = "https://api.stratz.com/graphql"

API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiN2Q1ZTQ5NjQtNTUyMC00MTUxLTlkYWYtMWM4YmNlZmQxY2YyIiwiU3RlYW1JZCI6IjExMzE0OTgzMTAiLCJuYmYiOjE2OTI4NDcxNTMsImV4cCI6MTcyNDM4MzE1MywiaWF0IjoxNjkyODQ3MTUzLCJpc3MiOiJodHRwczovL2FwaS5zdHJhdHouY29tIn0.hy_J0sxfdHDq6VBJkv6i1wlU5gry1L-_TjeAjg3hpvI"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


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

def generate_training_data():

    print("Generating Training Data")

    # note the url is 'graphql' and not 'graphiql'

    #wards?
    #items?
    #agi/str/int


    hero_query = """
    query {
        heroStats {
            stats{
            heroId
            apm
            casts
            abilityCasts
            kills
            deaths
            assists
            networth
            goldPerMinute
            xp
            cs
            dn
            neutrals
            heroDamage
            towerDamage
            physicalDamage
            magicalDamage
            pureDamage
            disableCount
            disableDuration
            stunCount
            stunDuration
            slowCount
            slowDuration
            healingSelf
            healingAllies
            invisibleCount
            runePower
            runeBounty
            supportGold
            level
            campsStacked
            ancients
            goldLost
            goldFed
            weakenCount
            weakenDuration
            physicalItemDamage
            magicalItemDamage
            healingItemSelf
            healingItemAllies
            attackDamage
            castDamage
            damage
            kDAAverage
            
            
            }
            
        }
    
    }
    """

    print(hero_query)

    r = requests.post(API_URL, json={"query":hero_query}, headers=HEADERS)
    data = r.json()

    df = pd.json_normalize(data['data']['heroStats']['stats'])
    df.to_csv('hero_data.csv', index=False)

#does not work for patch specific yet
def generate_matchIDs(n, patch=None):

    print("Generating MatchODS")

    oldest = 6079846053
    youngest = 7531641143
    my_range = youngest - oldest
    match_list = []


    for _ in range(n):
        match_list.append(int(np.random.rand() * my_range + oldest))

    return match_list


#returns true/false for single, array for mutliple
def check_real_game(match_id):
    print(f"Checking match id for realistic properties {match_id}")
    valid_lobby_types = ["UNRANKED", "TOURNAMENT", "RANKED"]

    if isinstance(match_id, int): 
        match_id_query = """
            query GetMatch($matchId: Long!) {
            match(id: $matchId) {
                durationSeconds
                numHumanPlayers
                lobbyType
            }
        }
        """

        variables = {
            "matchId": match_id
        }
    elif isinstance(match_id, list):
        match_id_query = """
            query GetMatches($matchIds: [Long!]!) {
            matches(ids: $matchIds) {
                id
                durationSeconds
                numHumanPlayers
                lobbyType
            }
        }
        """
        variables = {
            "matchIds": match_id
        }

    req_data = {"query": match_id_query, "variables": variables}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=req_data)  # Use json parameter instead of data for JSON payload
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


    if isinstance(match_id, int):
        df = pd.json_normalize(data['data']['matches'])

        # Filtering based on numHumanPlayers
        real_game = (df["numHumanPlayers"].iloc[0] == 10) and (df["lobbyType"].iloc[0] in valid_lobby_types)

        return real_game

    #need permissions
    elif isinstance(match_id, list):  
        print(data)
        df = pd.json_normalize(data['data']['match'])

        # Filtering based on numHumanPlayers
        df["realGame"] = (df["numHumanPlayers"] == 10) & (df["lobbyType"].isin(valid_lobby_types))

        return df["realGame"]
    else:
        print("Error, if you've gotten here something has seriously gone wrong")
        return None

