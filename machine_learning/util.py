import numpy as np
import requests
import json
import csv
import pandas as pd
from bs4 import BeautifulSoup

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

        # Extract player numbers as integers
        player_numbers = [int(link['href'].split("/")[-1]) for link in soup.find_all('a', href=lambda href: href and "/players/" in href and href.split("/")[-1].isdigit())]

        data = {'player_ids': player_numbers}

        # Store the list of player numbers as JSON
        with open('popular_players.json', 'w') as json_file:
            json.dump(data, json_file)

        return player_numbers
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)

def get_last_games(player_id):
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
        "request": {"skip": 0, "take": 40}
    }

    req_data = {"query": match_id_query, "variables": variables}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=req_data)  # Use json parameter instead of data for JSON payload
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"