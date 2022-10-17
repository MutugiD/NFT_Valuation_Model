import os
from ratelimit import limits
from ratelimit.decorators import sleep_and_retry
import requests
import time
import pymongo
from pymongo import MongoClient
CONNECTION_STRING = ""
client = MongoClient(CONNECTION_STRING)
db = client["nft"]
api_key = ''
db = client["nft"]

###inputs 
collection = "boredapeyachtclub"
contract_address = "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D"
m=1000 
n = 1041 
###
assets = db[collection]

def call_api(url, params):
    """Retrieve asset data"""
    headers = {
             "Accept": "application/json",
              "X-API-KEY": "b035c1c4866040108a7ea7ee5d19b7f6" }
    response = requests.get(url=url, params=params, headers=headers)
    time.sleep(0.18)
    if response.status_code !=200:
        print('API response: {}'.format(response.status_code))
        time.sleep(30)
        response = requests.get(url=url, params=params, headers=headers)
    return response
   
def create_event_params(token_id: str) -> dict:
    """Create query parameters for assets request"""
    return {
        "token_id": token_id,
        "asset_contract_address": contract_address,
        # "offset": 0,
        "limit": 100,
        "event_type": "successful" }

def process_many(token_ids: list, base_url: str, event:str) -> None:
    global collection
    for id in token_ids:
        fn = f'{db}.events/{id}.json'.format(db = collection)
        resp = call_api(events_url, params=create_event_params(token_id=1))
        data = resp.json()
        old = {'asset_events': 1}
        new = {"$set": {'asset_events': 1}}
        events.update_one(old, new)
        print(f'Processed {fn}!')
    print("Done")
    

events = db[collection]
events_url = "https://api.opensea.io/api/v1/events"
token_ids = [f'{i}' for i in range(m, n)]
process_many(token_ids=token_ids, base_url=events_url, event = events)
