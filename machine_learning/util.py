import numpy as np
import requests
import json
import csv
import pandas as pd




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



    # note the url is 'graphql' and not 'graphiql'
    url = "https://api.stratz.com/graphql"
    api_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiN2Q1ZTQ5NjQtNTUyMC00MTUxLTlkYWYtMWM4YmNlZmQxY2YyIiwiU3RlYW1JZCI6IjExMzE0OTgzMTAiLCJuYmYiOjE2OTI4NDcxNTMsImV4cCI6MTcyNDM4MzE1MywiaWF0IjoxNjkyODQ3MTUzLCJpc3MiOiJodHRwczovL2FwaS5zdHJhdHouY29tIn0.hy_J0sxfdHDq6VBJkv6i1wlU5gry1L-_TjeAjg3hpvI"
    headers = {"Authorization": f"Bearer {api_token}"}


    #gpm -> feature expand
    #wards?
    #items?
    #agi/str/int


    hero_query = """
    {
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

    r = requests.post(url, json={"query":hero_query}, headers=headers)
    data = r.json()

    # Use pandas to normalize the JSON data into a dataframe
    df = pd.json_normalize(data['data']['heroStats']['stats'])
    df.to_csv('hero_data.csv', index=False)
    # Print the dataframe
    print(df)



def generate_matchIDs():

    


    match_list = []





    return match_list