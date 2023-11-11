import requests

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

print(r.text)