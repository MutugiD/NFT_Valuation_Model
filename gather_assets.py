##asset and collection aggregated
import os
from ratelimit import limits
from ratelimit.decorators import sleep_and_retry
import requests
import time

key = 'b035c1c4866040108a7ea7ee5d19b7f6'
m = 1
n = 9500
collection = "onchainmonkey"
contract_address = "0x960b7a6bcd451c9968473f7bbfd9be826efd549a" 

# @sleep_and_retry
# @limits(calls=200, period=sleep_time)
def call_api(url, params):
    """Retrieve asset data"""
    headers = {"X-API-KEY": key}
    response = requests.get(url=url, params=params, headers=headers)
    time.sleep(0.18)
    if response.status_code !=200:
        print('API response: {}'.format(response.status_code))
        time.sleep(30)
        response = requests.get(url=url, params=params, headers=headers)
    return response

##collection info for each asset
def create_asset_params(token_id: str) -> dict:
    """Create query parameters for assets request"""
    return {
        "token_ids": token_id,
        
        "asset_contract_address": contract_address, 
        "order_direction": "desc",
        "offset": 0,
        "limit": 1,
        "collection": collection }

def write_to_json_file(fn: str, data: str) -> None:
    with open(fn, 'wb') as f:
        f.write(data)


def process_many(token_ids: list, files: list, base_url: str) -> None:
    for id in token_ids:
        fn = f'asset_events/assets/onchain-monkey/{id}.json'
        if not fn in files:
            resp = call_api(base_url, params=create_asset_params(token_id=id))
            data = resp.content
            write_to_json_file(fn=fn, data=data)
            print(f'Processed {fn}!')
    print("Done")


start_time = time.time()
assets_url = "https://api.opensea.io/api/v1/assets"
files = [f'onchain-monkey/{f}' for f in os.listdir('./asset_events/assets/onchain-monkey')]
token_ids = [f'{i}' for i in range(m, n+1)]
process_many(token_ids=token_ids, files=files, base_url=assets_url)
print("--- %s seconds ---" % (time.time() - start_time))