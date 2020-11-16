import pandas as pd 
from googletrans import Translator
import httpx


#Translate items
items_russia = pd.read_csv('items.csv')

trans = Translator()
items = []
for item in items_russia['item_name']:
    timeout = httpx.Timeout(5)
    translate = trans.translate(item,dest='en', timeout=timeout)
    items.append(translate.text)

#Create a new CSV with translation
items_to_df = pd.DataFrame(items, columns=['item_Name'])
not_trans = items_russia[['item_id', 'item_category_id']]    
frame = [not_trans, items_to_df]
df_concat = pd.concat(frame, axis=1, join='inner')
df_concat.to_csv('items_translated.csv')


#Translate shops 
shops_russia = pd.read_csv('shops.csv')

shops = []
for items in shops_russia['shop_name']:
    timeout = httpx.Timeout(5)
    translate = trans.translate(items,dest='en', timeout=timeout)
    shops.append(translate.text)
    
#Create new CSV with translated shop_names (easy way)
shops_to_df = pd.DataFrame(shops, columns=['shop_Name'])
shops_russia['shop_Name'] = shops_to_df
del shops_russia['shop_name']
shops_russia.to_csv('shops_translated.csv')
