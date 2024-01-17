import numpy as np
import requests
import json
import csv
import pandas as pd
from bs4 import BeautifulSoup
import time


API_URL = "https://api.stratz.com/graphql"

API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0IjoiN2Q1ZTQ5NjQtNTUyMC00MTUxLTlkYWYtMWM4YmNlZmQxY2YyIiwiU3RlYW1JZCI6IjExMzE0OTgzMTAiLCJuYmYiOjE2OTI4NDcxNTMsImV4cCI6MTcyNDM4MzE1MywiaWF0IjoxNjkyODQ3MTUzLCJpc3MiOiJodHRwczovL2FwaS5zdHJhdHouY29tIn0.hy_J0sxfdHDq6VBJkv6i1wlU5gry1L-_TjeAjg3hpvI"
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


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

def get_popular_players():
    url = "https://www.dotabuff.com/players/"
    my_header = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=my_header)

    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract player numbers as integers and their shown names
        player_info = [{'id': int(link['href'].split("/")[-1]), 'name': link.text} for link in soup.find_all('a', href=lambda href: href and "/players/" in href and href.split("/")[-1].isdigit())]

        data = {'players': player_info}

        unique_named_ids = set()
        filtered_players = []
        for player in data["players"]:
            player_id = player["id"]
            player_name = player["name"]
            
            if player_name != "" and player_id not in unique_named_ids:
                filtered_players.append(player)
                unique_named_ids.add(player_id)

        data["players"] = filtered_players

        # Store the list of player numbers and names as JSON
        with open('pp_list.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return player_info
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)

def get_last_matches_detailed(player_id, num_games):
    print(f"Getting info for player: {player_id}")
    match_id_query = """
    query PlayerMatchesSummary($request: PlayerMatchesRequestType!, $steamId: Long!) {
        player(steamAccountId: $steamId) {
            steamAccountId
            matches(request: $request) {
            ...MatchRowSummary
            players(steamAccountId: $steamId) {
                ...MatchRowSummaryPlayer
            }
            }
        }
    }

    fragment MatchRowBase on MatchType {
        id
        rank
        lobbyType
        gameMode
        endDateTime
        durationSeconds
        allPlayers: players {
            partyId
        }
        league {
            id
            displayName
        }
        analysisOutcome
    }

    fragment MatchRowBasePlayer on MatchPlayerType {
        steamAccountId
        heroId
        role
        lane
        level
        isVictory
        isRadiant
        partyId
    }

    fragment MatchRowSummary on MatchType {
        ...MatchRowBase
        bottomLaneOutcome
        midLaneOutcome
        topLaneOutcome
        pickBans {
            heroId
            isCaptain
        }
    }

    fragment MatchRowSummaryPlayer on MatchPlayerType {
        ...MatchRowBasePlayer
        imp
        award
        kills
        deaths
        assists
        item0Id
        item1Id
        item2Id
        item3Id
        item4Id
        item5Id
    }
    """

    variables = {
        "steamId": player_id,
        "request": {"skip": 0, "take": num_games}
    }

    req_data = {"query": match_id_query, "variables": variables}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=req_data)  # Use json parameter instead of data for JSON payload
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    
def get_last_matches_hero(player_id, num_games):
    print(f"Getting info for player: {player_id}")
    match_id_query =  """
    query PlayerMatchesHeroSummary($request: PlayerMatchesRequestType!, $steamId: Long!) {
        player(steamAccountId: $steamId) {
            matches(request: $request) {
      			id
                players(steamAccountId: $steamId) {
                    heroId
                }
            }
        }
    }
    """

    variables = {
        "steamId": player_id,
        "request": {"skip": 0, "take": num_games}
    }

    req_data = {"query": match_id_query, "variables": variables}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=req_data)  # Use json parameter instead of data for JSON payload
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

    match_data = {}
    match_data["m_id"] = data["data"]["player"]["matches"]
    return match_data
    

def popular_players_past_games(num_games, num_players = 10):
    with open('pp_list.json', 'r') as file:
        pp_dict = json.load(file)
    
    # Set the desired rate of API calls (200 calls per minute)
    calls_per_minute = 200
    seconds_per_call = 60 / calls_per_minute

    all_player_data = {}
    limit = 0
    for p_id in pp_dict["players"]:
        all_player_data[p_id["id"]] = get_last_matches_hero(player_id=p_id["id"], num_games=num_games)        
        time.sleep(seconds_per_call)

        limit += 1
        if limit >= num_players:
            break
    
    print("Writing to file")
    
    for match in all_player_data[p_id["id"]]["m_id"]:
        all_player_data[p_id["id"]] = match["players"][0]["heroId"]

    with open('pp_data_hero.json', 'w') as json_file:
            json.dump(all_player_data, json_file, indent=4)

    