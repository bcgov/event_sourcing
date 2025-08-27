print('Started generating data')

# %%
import pandas as pd
import numpy as np
import copy

# %%
headers = ['snapshot_date', 'type', 'colour', 'size', 'featured', 'id']

#%%
from pathlib import Path

dirname = Path(__file__).resolve().parent

# %%
items = [
    {'id': 1, 'snapshot_date': ' 2024-03-31', 'type': 'jacket', 'colour': 'blue', 'size': 4, 'featured': 'N'},
    {'id': 2, 'snapshot_date': ' 2024-03-31', 'type': 'shorts', 'colour': 'gray', 'size': 0, 'featured': 'Y'},
    {'id': 3, 'snapshot_date': ' 2024-03-31', 'type': 'jeans', 'colour': 'green', 'size': 12, 'featured': 'N'},
    {'id': 4, 'snapshot_date': ' 2024-03-31', 'type': 'long-sleeve shirt', 'colour': 'plaid', 'size': 10, 'featured': 'Y'},
    {'id': 111, 'snapshot_date': '2024-03-31', 'type': 'blouse', 'colour': 'pink', 'size': 6, 'featured': 'N'}
]  

# %%
snaps = {}

# %%
dates = [
    '2024-03-31',
    '2024-04-30',
    '2024-05-31',
    '2024-06-30',
    '2024-07-31',
    '2024-08-31',
    '2024-09-30'
]

# %%
for date in dates:
    snaps[date] = pd.DataFrame(copy.deepcopy(items))
    snaps[date]['snapshot_date'] = date

# %%
for date in dates[2:]:
    snaps[date].loc[snaps[date].id == 1, 'featured'] = 'Y'

# %%
for date in dates[3:]:
    snaps[date].loc[snaps[date].id == 2, 'colour'] = 'sage'

# %%
for date in dates[3:]:
    i = snaps[date][snaps[date].id == 4].index
    snaps[date] = snaps[date].drop(i).reset_index(drop=True)

# %%
for date in dates[3:]:
    snaps[date] = pd.concat([snaps[date], pd.DataFrame({
        'id': [5, np.nan], 
        'snapshot_date': date, 
        'type': ['pants', 'socks'], 
        'colour': ['purple', 'gray'], 
        'size': [2, np.nan], 
        'featured': ['Y', np.nan]
    })]).reset_index(drop=True)

# %%
snaps[dates[4]].loc[len(snaps[dates[4]])] = {
    'id': 6, 
    'snapshot_date': dates[4], 
    'type': 'shirt', 
    'colour': 'blue', 
    'size': 8, 
    'featured': 'Y'
}

# %%
snaps[dates[4]].loc[len(snaps[dates[4]])] = {
    'id': np.nan, 
    'snapshot_date': dates[4], 
    'type': 'hat', 
    'colour': 'orange', 
    'size': np.nan, 
    'featured': np.nan
}

# %%
snaps[dates[2]].loc[len(snaps[dates[2]])] = {
    'id': 99, 
    'snapshot_date': dates[2], 
    'type': 'skirt', 
    'colour': 'striped', 
    'size': np.nan, 
    'featured': np.nan
}

# %%
for date in dates[2:]:
    snaps[date].loc[len(snaps[date])] = {
        'vin': 99, 
        'snapshot_date': date, 
        'type': 'scarf', 
        'colour': 'beige', 
        'size': np.nan, 
        'featured': np.nan
    }

# %%
snaps[dates[4]].loc[len(snaps[dates[4]])] = {
    'id': 111, 
    'snapshot_date': dates[4], 
    'type': 'sweater', 
    'colour': 'striped', 
    'size': 10, 
    'featured': 'N'}

# %%
snaps[dates[4]].loc[len(snaps[dates[4]])] = {
    'id': 111, 
    'snapshot_date': dates[4], 
    'type': 'gloves', 
    'colour': 'gray', 
    'size': np.nan, 
    'featured': 'N'}

# %%
i = snaps[dates[3]].loc[snaps[dates[3]].id == 1].index
snaps[dates[3]] = snaps[dates[3]].drop(i)

# %%
snaps[dates[4]].rename(index={0:10},inplace=True)

# %%
snaps[dates[4]] = snaps[dates[4]].sort_index()

# %%
snaps[dates[0]] = snaps[dates[0]].drop(columns='featured')

# %%
snaps[dates[0]]['location'] = ['Vancouver', 'Victoria', 'Victoria', 'Vancouver', np.nan] 

# %%
snaps[dates[5]] = snaps[dates[5]].loc[~snaps[dates[5]]['id'].isin([111, 99, np.nan])].reset_index(drop=True)

# %%
snaps[dates[6]] = snaps[dates[5]].copy()

# %%
snaps[dates[6]]['snapshot_date'] = dates[6]

# %%
snaps[dates[6]].loc[len(snaps[dates[6]])] = {
    'vin': 111, 
    'snapshot_date': dates[6], 
    'type': 'vest', 
    'colour': 'cream', 
    'size': 16, 
    'featured': 'N'}

# %%
snaps[dates[6]].loc[len(snaps[dates[6]])] = {
    'vin': 99, 
    'snapshot_date': dates[6], 
    'type': 'coat', 
    'colour': 'brown', 
    'size': 2, 
    'featured': np.nan
}

# %%
for date in dates:
    snaps[date] = snaps[date].reset_index(drop=True)

# %%
for date in dates:
    snaps[date].to_csv(dirname / '.'.join([date, 'csv']), index=False)

print('Finished generating data')

