from pymongo import MongoClient
import os
from ratelimit import limits
from ratelimit.decorators import sleep_and_retry
import requests
import time
import pymongo
from pymongo import MongoClient
import json
from contextlib import suppress
CONNECTION_STRING = ""
client = MongoClient(CONNECTION_STRING)
api_key = ''

###inputs######
collection = "onchainmonkey"
contract_address = "0x960b7a6bcd451c9968473f7bbfd9be826efd549a"
m= 1
n = 9500
#########

api_time_sleep = 0.18  
throttle_cooldown = 30
def call_api(url, params):
    global api_key, api_time_sleep, throttle_cooldown 
    """Retrieve asset data"""
    headers = {
             "Accept": "application/json",
              "X-API-KEY": api_key}
    response = requests.get(url=url, params=params, headers=headers)
    time.sleep(api_time_sleep)
    #except json.decoder.JSONDecodeError as e:
    if response.status_code !=200:
        with suppress(json.decoder.JSONDecodeError): 
            print('API response: {}'.format(response.status_code))
            time.sleep(throttle_cooldown)
            response = requests.get(url=url, params=params, headers=headers)
    return response

#collection info for each asset
def create_asset_params(token_id: str) -> dict:
    """Create query parameters for assets request"""
    return {
        "token_ids": token_id,
        "asset_contract_address": contract_address, 
        "order_direction": "desc",
        "offset": 0,
        "limit": 1,
        "collection": collection }

def process_many(token_ids: list, base_url: str, asset) -> None:
    global collection
    for id in token_ids:
        fn = f'{db}.assets/{id}.json'.format(db = collection)
        resp = call_api(base_url, params=create_asset_params(token_id=id))
        data = resp.json()
        asset.insert_one(data)
        print(f'Processed {fn}!')
    print("Done")

db = client["nft"]
assets = db[collection]
assets_url = "https://api.opensea.io/api/v1/assets"
token_ids = [f'{i}' for i in range(m, n+1)]
process_many(token_ids=token_ids, base_url=assets_url, asset = assets)
start_time = time.time()
print("--- %s seconds ---" % (time.time () - start_time))
