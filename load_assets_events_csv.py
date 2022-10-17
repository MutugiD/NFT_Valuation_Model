# Module to load data from json files, extract desired fields and save to CSV

from this import s
import numpy as np
import pandas as pd
import json
import requests
from sqlalchemy import true

# base_url = "https://api.binance.com/api/v3"
# url = base_url + f"/avgPrice?symbol=ETHUSDT"
# usd_price = requests.get(url). json()['price']
m = 1
n = 9500

def load_json_data(fn: str) -> dict:
    """Load json file from directory"""
    with open(fn) as f:
        data = json.load(f)
        try: 
            if 'assets' in data:
                return data['assets'].pop()
            else:
                return data['asset_events']
        except: 
            return data

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
    global usd_price
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
        #d['USDPrice'] = d['LastSalePrice'] *float(usd_price)
        d['USDPrice'] = d['LastSalePrice'] * sale['usd_price']
        d['SaleDate'] = sale['created_date']

    for trait in traits:
        d[trait['trait_type']] = trait['value']
        d[trait['trait_type']+'Rarity'] = trait['trait_count']/10000
    return d

def create_pandas_df(m, n): 
    df = pd.DataFrame()
    x = range(m, n)
    for i in x:
        fn1 = f'asset_events/assets/bayc/{i}.json'
        fn2 = f'asset_events/events/bayc/{i}.json'
        asset_raw_data = load_json_data(fn=fn1)
        events_raw_data = load_json_data(fn=fn2)
        asset_formatted_data = create_formatted_dict(
            asset_raw_data['traits'], events=events_raw_data)
        df = df.append(asset_formatted_data, ignore_index=True)
    return df

m = 0
n = 9999
df = create_pandas_df(m,n )
df.index = np.arange(m, n)
# df_filtered = df.loc[df['LastSalePrice'] != 'NaN']  # can only train models on data that has had a sale
#df = df.reset_index(level=0)
df.to_csv('bayc.csv', index=True)
