import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, LassoCV, Lasso, Ridge
#from xgboost.sklearn import XGBRegressor
from sklearn.metrics import confusion_matrix
from typing import List
import joblib
import json
import uvicorn
import pickle 
from fastapi import FastAPI
from pydantic import BaseModel 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
class nft_inputs(BaseModel):
    contract_address: str
    token_id: list
@app.get('/')
async def home_call(): 
    return {"data": "Welcome to NFT"}


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/predict_nft')
async def predict_nft(nft:nft_inputs):
    nft_dict = nft.dict()
    address = nft_dict['contract_address']
    token_id =  nft_dict['token_id']

    
    def contract(address): 
        slug_dict = {"boredapeyachtclub":"0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d", 
                "clonex": "0x49cf6f5d44e70224e2e23fdcdd2c053f30ada28b", 
                "cool-cats-nft": "0x1a92f7381b9f03921564a437210bb9396471050c", 
                "mutant-ape-yacht-club": "0x60e4d786628fea6478f785a6d7e704777c86a7c6", 
                "cryptopunks":"0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb", 
                "psychedelics-anonymous-genesis": "0x75e95ba5997eb235f40ecf8347cdb11f18ff640b", 
                "grayboys":"0x8d4100897447d173289560bc85c5c432be4f44e4", 
                "world-of-women-nft":"0xe785e82358879f061bc3dcac6f0444462d4b5330", 
                "the-crypto-chicks": "0x1981cc36b59cffdd24b01cc5d698daa75e367e04", 
                "onchainmonkey":"0x960b7a6bcd451c9968473f7bbfd9be826efd549a"} 
        slug_df = pd.DataFrame(slug_dict.items(), columns =['slug', 'address'])
        slug = slug_df.loc[slug_df['address'] == address, 'slug'].iloc[0]
        return slug 
    pickle_in = open("{}.pkl".format(contract(address)),"rb")
    model = pickle.load(pickle_in)
    raw = pd.read_csv("{}.csv".format(contract(address)), #
                  parse_dates=['SaleDate']).rename(columns={'Unnamed: 0': 'TokenId'})
    df = raw.loc[(raw['USDPrice'] != 0) & (~raw['LastSalePrice'].isna())]
    df = df.sort_values(['SaleDate']).reset_index(level=0, drop=True)  # sort by sale date and time
    ts = df.groupby('SaleDate').agg({'USDPrice': 'mean'})
    ts_outrm = ts.loc[ts['USDPrice'] < 10e5]
    log_price = np.log(ts_outrm['USDPrice'])
    rolling_median = log_price.rolling(window=14).median()
    df['LogUSDPrice'] = np.log(df['USDPrice'])
    mean = df['LogUSDPrice'].ewm(span=14).mean()  # exponentially weighted moving average with 14 point window
    std = df['LogUSDPrice'].ewm(span=14).std()
    mean_plus_std = mean + 1.7*std  # 1.7 worked well
    is_outlier = df['LogUSDPrice'] > mean_plus_std
    df['Outlier'] = 1
    df.loc[is_outlier, 'Outlier'] = -1
    dfo, dfi = df[is_outlier].copy(), df[~is_outlier].copy()

    dfs = dfo, dfi
    for data in dfs:
        rolling_median = data['LogUSDPrice'].rolling(window=7, min_periods=1).median()
        ewm = data['LogUSDPrice'].ewm(span=14).mean()
        data['LogUSDPriceEWM'] = (rolling_median + ewm) / 2
        # Percentage Extension from the Exponential Weighted Moving Average
        data['PctExtensionEWM'] = data.apply(lambda x: (x['LogUSDPrice'] - x['LogUSDPriceEWM']) / x['LogUSDPriceEWM'], axis=1)

    def rolling_periods(data): 
        dfs = dfo, dfi
        for data in dfs:
            rolling_median = data['LogUSDPrice'].rolling(window=7, min_periods=1).median()
            ewm = data['LogUSDPrice'].ewm(span=14).mean()
            data['LogUSDPriceEWM'] = (rolling_median + ewm) / 2
            # Percentage Extension from the Exponential Weighted Moving Average
            data['PctExtensionEWM'] = data.apply(lambda x: (x['LogUSDPrice'] - x['LogUSDPriceEWM']) / x['LogUSDPriceEWM'], axis=1)
        return dfs

    def rarity_columns(data): 
        rarity_cols = [c for c in data.columns if 'Rarity' in c]  # get all numeric rarity related cols

        def fill_trait_na(col): 
            fill_value = 1 - col.dropna().unique().sum()
            return fill_value
        for data in rolling_periods(data):
            data[rarity_cols] = data[rarity_cols].apply(lambda x: x.fillna(fill_trait_na(x)))
        return  rarity_cols
    #cols = rarity_processing(data)
    def traits(data): 
        traits = []
        for cols in rarity_columns(data): 
            col = cols.split("Rarity")
            traits.append(col[0])
        for data in dfs:
            data[traits] = data[traits].fillna('None')
        return traits 
    #rarity_cols = rarity_columns(data)

    def top_rarity(df, n=2): 
        cols = [x for x in df.columns if 'Rarity' in x]
        value = pd.DataFrame()
        for col in cols:
            value[col] = 1/df[col]
        sum_total = value.loc['Total'] = value.sum()
        top_two = sum_total.nlargest(n)
        top_traits = top_two.index
        return top_traits 
    def rare_traits(df): 
        rarest_trait1 = df[top_rarity(df, n=2)[0].split("Rarity")[0]]
        rarest_trait2 = df[top_rarity(df, n=2)[1].split("Rarity")[0]]
        traits_df = pd.DataFrame({'rarest_trait1':rarest_trait1, 'rarest_trait2':rarest_trait2})

        countnames1 = {}
        for name in rarest_trait1:
            if name in countnames1:
                countnames1[name] += 1
            else:
                countnames1[name] = 1
        rare_trait_1 = min(countnames1, key = countnames1.get)

        countnames2 = {}
        for name in rarest_trait2:
            if name in countnames2:
                countnames2[name] += 1
            else:
                countnames2[name] = 1
        rare_trait_2 = min(countnames2, key = countnames2.get)
        
        return  rare_trait_1,  rare_trait_2 
    rarity_1 = top_rarity(df)[0]
    rarity_2 = top_rarity(df)[1]
    trait_1 = top_rarity(df)[0].split("Rarity")[0]
    trait_2 = top_rarity(df)[1].split("Rarity")[0]
    rare_trait_1 = rare_traits(df)[0]
    rare_trait_2 = rare_traits(df)[1]

    def has_two_less_1pct(row, cols):
        rarity = row[cols].values
        n = len(rarity[np.where(rarity < 0.01)])
        if n> 1:
            return 1
        return 0
    def rarity_1_or_rarity_2(row):
        if row[rarity_1] < 0.01 or row[rarity_2] < 0.01:
            return 1
        return 0

    def trait_1_or_trait_2(row):
        if row[trait_1] == rare_trait_1  or  row[trait_2] == rare_trait_2:
            return 1
        return 0

    def find_matches(row, categories):
        traits = row[categories].values
        keywords = []
        for trait in traits:
            split = trait.split(' ')
            for word in split:
                keywords.append(word)
        if 'None' in keywords:
            keywords.remove('None')
        counts = Counter(keywords)
        most_common = counts.most_common(1)
        matches = most_common[0][1] - 1
        if matches:
            return 1
            return 0

    dfs = rolling_periods(df)
    rarity_cols = rarity_columns(df)
    traits = traits(df)
    for data in dfs:
            data['HasTwoLess1Pct'] = data.apply(lambda x: has_two_less_1pct(x, rarity_cols), axis=1)
            data['Rarity1OrRarity1'] = data.apply(rarity_1_or_rarity_2, axis=1)
            data['Trait1OrTrait2'] = data.apply(trait_1_or_trait_2, axis=1)
            data['HasMatches'] = data.apply(lambda x: find_matches(x, traits), axis=1)

    for data in dfs:
        data.loc[(data['HasTwoLess1Pct']==1) | (data['Rarity1OrRarity1']==1) | (data['Trait1OrTrait2']==1), 'OutlierRule'] = -1
        data['OutlierRule'] = data['OutlierRule'].fillna(1)

    def outlier_encoding(dfo): 
        categorical_feature_cols = traits 
        clf_features = ['HasTwoLess1Pct','Rarity1OrRarity1', 'Trait1OrTrait2', 'HasMatches']
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(dfo[categorical_feature_cols]).toarray())
        enc_df.index = list(enc_df.index)  # convert index to int64 index
        enc_df.columns = enc.get_feature_names_out()
        dfp = dfo.reset_index(drop=True).merge(enc_df.reset_index(drop=True), left_index=True, right_index=True)
        dfo = dfp.drop(columns=categorical_feature_cols)
        return dfo

    def inlier_encoding(dfi): 
        categorical_feature_cols = traits 
        clf_features = ['HasTwoLess1Pct','Rarity1OrRarity1', 'Trait1OrTrait2', 'HasMatches']
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(dfi[categorical_feature_cols]).toarray())
        enc_df.columns = enc.get_feature_names_out()
        dfp = dfi.reset_index(drop=True).merge(enc_df.reset_index(drop=True), left_index=True, right_index=True)
        dfi = dfp.drop(columns=categorical_feature_cols)
        return dfi

    dfo = outlier_encoding(dfo)
    dfi = inlier_encoding(dfi)
    df = pd.concat([dfi, dfo])
    df.fillna(0,inplace=True)

    def token_train_test(df, token_id):  
        features = [feature for feature in df.columns if '_' in feature]
        features.extend(['HasMatches', 'NumberOfSales'])
        token = df.loc[df['TokenId'].isin(token_id)]
        df = df[df['TokenId'].isin(token_id) == False]
        X = df.loc[:, features+['OutlierRule', 'Outlier']]
        y = df['PctExtensionEWM']
        x_token = token.loc[:, features+['OutlierRule', 'Outlier']]
        y_token = token['PctExtensionEWM']
        return x_token, y_token

    x_token, y_token  = token_train_test(df, token_id)

    def predict(X, models):
        outliers = X['OutlierRule'] == -1
        X_o = X.loc[outliers].drop(columns=['OutlierRule', 'Outlier'])
        X_i = X.loc[~outliers].drop(columns=['OutlierRule', 'Outlier'])
        y = np.empty(X.shape[0])  # store combined predictions
        # predict target values separately accorindg to the two models
        
        if X_o.shape[0]:
            y_o_pred = models['o'].predict(X_o)
            y[outliers] = y_o_pred
        if X_i.shape[0]:
            y_i_pred = models['i'].predict(X_i)
            y[~outliers] = y_i_pred
        return y
    def predict_mapping(y_test): 
        y_actual, y_pred = y_test, predict(x_token, model)
        predict_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred})
        predicted_merged = df.merge(predict_df, how='left', left_index=True, right_index=True)
        predicted_merged = predicted_merged.loc[~predicted_merged['Predicted'].isna()]
        cols = predicted_merged.columns[0:]
        pm = predicted_merged.loc[:, ['TokenId', 'SaleDate', 'LogUSDPriceEWM', 'Actual', 'Predicted', 'USDPrice', 'PctExtensionEWM']]
        pm['PredictedUSDPrice'] = np.exp(pm['LogUSDPriceEWM'] * (1 + pm['Predicted']))
        pm['ContractAddress'] = address
        pm['Accuracy'] = 1 - abs(pm['PredictedUSDPrice'] - pm['USDPrice']) / ((pm['PredictedUSDPrice'] + pm['USDPrice']) / 2)
        return pm[['TokenId', 'ContractAddress', 'USDPrice', 'PredictedUSDPrice']]
    
    token = predict_mapping(y_token)
    token = token.loc[token['TokenId'].isin(token_id)]
    result = token.to_json(orient="records")
    return json.loads(result)
    
if __name__ == '__main__':
   uvicorn.run(app, host='127.0.0.1', port=8000)


