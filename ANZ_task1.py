# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # ANZ Data Virtual Internship - Task 1

# %%
# Imports
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd 
from shapely.geometry import Point 
from geopandas import GeoDataFrame
import plotly_express as px

#Pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# seaborn settings
sns.set_style("darkgrid")

# %% [markdown]
# ## Data Importing and Exploratory Data Analysis

# %%
# read in csv file 
data = pd.read_csv('ANZ_synthesised_transaction_dataset.csv')
# display head
print(data.head())


# %%
print(data.shape)


# %%
print(data.describe()) #Perform basic summary stats on numeric columns

# %% [markdown]
# Noting above that the pd.describe displays float and integer objects only, this means many other columns are encoded as objects. The below code calls upon df.dtypes to determine the types of values in each column.

# %%
print(data.dtypes)


# %%
print(data.nunique()) # Determines the number of unique values per column

# %% [markdown]
# The above code shows us that as per the description we do indeed have 100 customers data based off 100 unique values for the accounts column.

# %%
print(data.isnull().sum()) # counts the number of null values for each column


# %%
# Check the percentage of missing values per column
print("Percentage of missing values:")
print()
for column in data.columns:
    print(f'Column {column} has'
          f' {100 * sum(data[column].isna()) / len(data):.2f}%'
          f' missing values')

# %% [markdown]
# Based off the above code, we have significant numbers of null values in the dataset. However considering this is transactional data:
# 
#     1. Not all payments were via Bpay - hence we have a lack of values in bpay_biller_code
#     2. We have 4326 missing values in the card_present_flag, merchant_id, merchant_suburb, merchant_state and   merchant_long_lat columns. This could be due to the card not being present at the time of transaction (online or manual purchases) or for another reason entirely.
#     3. We are missing a lot of data in the merchant_code column (~92%). This could be due to most transctions (~92%) are not Bpay transactions and will hence not have a merchant code. As such we should remove this column alongside the bpay_biller_code column.
# %% [markdown]
# ## Data Cleaning

# %%
# assign to clean data frame variable
data_clean = data

# split up lat_long column into lat and long for ease of plotting later
data_clean[['long','lat']] = data_clean['long_lat'].str.split(' ', expand=True)

# split up merchant long_lat into lat and long for ease of plotting later
data_clean[['merchant_long','merchant_lat']] = data_clean['merchant_long_lat'].str.split(' ', expand=True)
#data_clean.head() # Sanity check


# %%
# drop columns missing signifcant amounts of data / unneeded columns:

# 1. merchant_code - missing data (see above)
# 2. currency - all in AUD in this dataset (based off unique values)
# 3. country - all in Australia in this dataset (based off unique values)
# 4. long_lat - not needed after split above
# 5. merchant_long_lat - not needed after split above
data_clean = data_clean.drop(['merchant_code','currency', 'country', 'long_lat','merchant_long_lat', 'bpay_biller_code'], axis=1)


# %%
# Change dtypes to numeric for all latitude and longitude columns
data_clean = data_clean.astype({'long':'float64', 'lat':'float64', 'merchant_long':'float64', 'merchant_lat':'float64'})

# ensure that the date column is a datetime object
data_clean['date'] = pd.to_datetime(data_clean['date'], format= '%d/%m/%y')

# extract day of week from date and add to df - represented by number
data_clean['weekday'] = data_clean['date'].dt.dayofweek

# create dictonary of name of days based off pandas dayofweek function
day_of_week_names={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
# assign and map weekday column to our dictonary of day names
data_clean['weekday'] = data_clean['date'].dt.dayofweek.map(day_of_week_names)


# %%
# Sort data by date
data_clean.sort_values(by=['date'], inplace=True)


# %%
print(data_clean.describe())

# %% [markdown]
# Noting the above that in the 'lat' and 'long' columns we have large values of 255 degrees and -573 degrees respectively. These values for latitude and longitude are not possible.

# %%
# Investigate the above by locating rows where -573 is the value in the 'lat' column
data_clean.loc[data_clean['lat'] == -573].head() # note --> .head() used here simply to trim output as there is a number of rows - remove .head() to investigate all rows as required.

# %% [markdown]
# From the above output we can see that our unsual value for longitude of 255 degrees appears to be for the same customer (Daniel) whos latitude value is -573. Whilst the remaining data could be accurate, we will drop these rows as this will throw off our visualisations later on.

# %%
# assign the indexs of the affending rows
indexnames_daniel_latlong = data_clean[data_clean['lat']==-573].index
# drop rows with the corresponding index from data_clean
data_clean.drop(indexnames_daniel_latlong, inplace=True)


# %%
print('Therefore a loss of {} rows of data.'.format(data.shape[0]-data_clean.shape[0]))

# %% [markdown]
# ## Insights into the dataset

# %%
# Average transaction amount
print('The average transaction amount is ${:.2f} and the median transaction amount is ${:.2f}'.format((data_clean['amount'].mean()), data_clean['amount'].median()))

print() # creating a space in output
# Average number of transaction per customer over the 3 month period
print('The average number of transaction per person over the 3 month period is {:.2f} and the median number of transactions is {:.2f}'.format(data_clean['customer_id'].value_counts().mean(), data_clean['customer_id'].value_counts().median()))

print() # creating a space in output
# Average balance in an ANZ account across the 3 month period
print('The average balance in an ANZ account across the 3 month period is ${:.2f} and the median balance is ${:.2f}'.format(data_clean['balance'].mean(), data_clean['balance'].median()))

# %% [markdown]
# The above output is showing us that the average transaction amount has been greatly affected by outliers, with an average transaction amount of *$187 and a median value of *$29. 
# 
# The code above also shows that there is a relatively small difference in the average and median number of transactions per customer across the 3 month period, thus we can conclude that the number of transactions per customer was evenly distributed in this time period.
# 
# The average balance in an ANZ account at the time of transaction also differed largely compared to the median balance in an account. The average balance across the time period was *$14707, and the median balance was *$6432, indicating the presence of outliers in the dataset (ie. accounts with large balances).
# %% [markdown]
# ### Transaction volume

# %%
# count the number of transaction per day - store as df
date_transactions_counting = data_clean.groupby('date').count()


# %%
# seaborn lineplot using grouped dataframe of transactions and dates to form transaction volume
sns.lineplot(date_transactions_counting.index, date_transactions_counting['customer_id'])
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Num. of Transactions')
plt.title('Transaction volume of ANZ accounts from 1/8/18 to 1/11/18')
plt.show()


# %%
sns.lineplot(x='date', y='amount', data=data_clean, hue='gender')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Dollar value of transactions ($)')
plt.title('Dollar value of transactions of ANZ accounts from 1/8/18 to 1/11/18 by gender')
plt.show()

# %% [markdown]
# We can see a trend here, lets break this down to examine transactions across a week.
# %% [markdown]
# ### Spending across a typical week

# %%
# sns lineplot
sns.lineplot(x='weekday', y='amount', data=data_clean, hue='gender')
plt.xticks(rotation=45)
plt.xlabel('Weekday')
plt.ylabel('Dollar value of transactions ($)')
plt.title('Dollar value of transactions of ANZ accounts across a typical week')
plt.show()

# %% [markdown]
# Interestingly we can see that transactions for both males and females increase at the begining of the work week (Monday) and then decrease on a Tuesday. Males in the datset tend to then increase spending until Friday before spedning drops on a weekend. Females in the dataset tend to spend less on a Thursday before increasing spending on a Friday.
# %% [markdown]
# ### Spending by State

# %%
# create new df of the sum of transactions grouped by the merchants state
merchant_groupby_state = data_clean.groupby(['merchant_state'])['amount'].sum().reset_index()
# sort by largest first
merchant_groupby_state = merchant_groupby_state.sort_values('amount', ascending=False)
merchant_groupby_state.head(8) # 8 states and territories


# %%
# visualise the above dataframe
sns.barplot(x='amount', y='merchant_state', data=merchant_groupby_state)
plt.xticks(rotation=45)
plt.xlabel('Dollars ($AUD')
plt.ylabel('Merchant state')
plt.title('Spending by merchant state of 100 ANZ customers from 1/8/18 to 1/11/18')
plt.show()

# %% [markdown]
# This is in line with the population rates of each state and territory of Australia. 
# 
# Source:(https://en.wikipedia.org/wiki/States_and_territories_of_Australia#States_and_territories)

# %%
# create new df of the sum of transactions by merchant state per date - ie. the total of transactions (AUD) done in a state per day
merchant_groupby_state_date = data_clean.groupby(['date','merchant_state'])['amount'].sum().reset_index()
# sort by largest first
merchant_groupby_state_date = merchant_groupby_state_date.sort_values('amount', ascending=False)
merchant_groupby_state_date.head()


# %%
# visualise the above dataframe
sns.lineplot(x='date', y='amount', data=merchant_groupby_state_date, hue='merchant_state')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Dollar value of transactions ($)')
plt.title('Dollar value of transactions of ANZ accounts from 1/8/18 to 1/11/18 by merchant state')
plt.show()

# %% [markdown]
# The above shows that the largest 3 states by population (NSW, VIC, QLD) have the highest spending over time. Notably there was a significant increase in spending in NSW on the 29th of September 2018 and in QLD on the 21st of October 2018

# %%
# new df of the sum of transactions aggregated by merchant suburb
merchant_groupby_suburb = data_clean.groupby(['merchant_suburb'])['amount'].sum().reset_index()
merchant_groupby_suburb = merchant_groupy_suburb.sort_values('amount', ascending=False)
merchant_groupby_suburb.head()

# %% [markdown]
# The above dataframe shows that the majority of spending on these ANZ accounts is occuring in major CBD's and surrounding suburbs.
# %% [markdown]
# ### Geographical plotting of Transactions

# %%
# using point from shaply - zip together long and lat columns to create our coorinates
transactions_geographical_lat_long = [Point(xy) for xy in zip(data_clean['long'], data_clean['lat'])]
# create geo pandas df using above zipped coordinates and link with data_clean
dataframe_geopandas = GeoDataFrame(data_clean, geometry= transactions_geographical_lat_long)
# create plotly express scatter geo plot
fig = px.scatter_geo(dataframe_geopandas, lat=dataframe_geopandas.geometry.y, lon=dataframe_geopandas.geometry.x, hover_name='amount')
fig.update_geos(fitbounds='locations')  # ensures zoom level is on Aus
fig.show()


# %%



