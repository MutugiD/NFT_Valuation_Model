import numpy as np
import pandas as pd
import json
import requests
import time
import pymongo
from pymongo import MongoClient
CONNECTION_STRING = ""
client = MongoClient(CONNECTION_STRING)
db = client["nft"]

#'grayboys', 'boredapeyachtclub' (empty list), 'onchainmonkey', 
###inputs######
collection = "onchainmonkey"
m= 1
n = 9500
#########
####
assets = db[collection]
events = db[collection]

def get_max_sale(sales: dict) -> dict:
    """Return the sale metadata which had the highest USD transaction value"""
    greatest_sale: dict
    greatest_sale_price = -1
    for sale in sales.values():
        sale_price = sale['total_price'] / 10**sale['decimals']
        usd_value = sale_price * sale['usd_price']
        if usd_value > greatest_sale_price:
            greatest_sale = sale
            greatest_sale_price = usd_value
    return greatest_sale

def create_formatted_dict(traits: dict, events: dict) -> dict:
    """Transform dictionary format"""
    sales = {}
    d = {}
    for i, event in enumerate(events):
        # gather only data relating to the most recent sale of the nft
        sales[f'sale_{i+1}'] = {'total_price': float(event['total_price']),
                                'decimals': float(event['payment_token']['decimals']),
                                'token_name': event['payment_token']['name'],
                                'usd_price': float(event['payment_token']['usd_price']),
                                'num_sales': len(events),
                                'created_date': event['created_date']}

    if not sales:
        d['LastSalePrice'] = 'NaN'
    else:
        sale = get_max_sale(sales)
        d['LastSalePrice'] = sale['total_price'] / 10**sale['decimals']
        d['LastSaleToken'] = sale['token_name']
        d['NumberOfSales'] = sale['num_sales']
        d['USDPrice'] = d['LastSalePrice'] * sale['usd_price']
        d['SaleDate'] = sale['created_date']

    for trait in traits:
        d[trait['trait_type']] = trait['value']
        d[trait['trait_type']+'Rarity'] = trait['trait_count']/10000
    return d

def create_pandas_df(n): 
    global assets
    df = pd.DataFrame()
    query = [x for x in assets.find({})]
    asset_data = [x for x in assets.find({}, {'_id': 0, 'assets': 1})]
    asset_data = list(filter(None, asset_data))
    event_data = [x for x in assets.find({}, {'_id': 0, 'asset_events': 1})]
    event_data = list(filter(None, event_data))
    for i in range(n):
        try: 
            asset_raw_data = asset_data[i].pop('assets').pop()
            events_raw_data = event_data[i].pop('asset_events')
        except:
             pass 
            
        asset_formatted_data = create_formatted_dict(
            asset_raw_data['traits'], events=events_raw_data)
        
        df = df.append(asset_formatted_data, ignore_index=True)
    return df 

df = create_pandas_df((n+1)- m)
df.index = np.arange(m, (n+1))
df.to_csv('{}.csv'.format(collection))
