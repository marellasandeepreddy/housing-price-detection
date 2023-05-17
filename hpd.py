import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import chardet#for encoding
import warnings# to avoid the warnings
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns',0)


#Let's see which encoding we have to apply.
with open("new.csv","rb") as f:
    result=chardet.detect(f.read(100000))
print(result)

#so,we have to apply GB2312 encoding.
data=pd.read_csv("new.csv",encoding="GB2312")

data.head()

data.shape

df0=data.copy()

data.columns

data.info()

data.isnull().sum()

#Let's Visualize the missing value
sns.heatmap(data.isnull(),yticklabels=False,cbar=False)

#Drop 'DOM' Columns
data.drop(columns=['DOM'],axis=1,inplace=True)
In [12]:
data.shape

data.isnull().sum()

data['buildingType'].fillna(data.buildingType.mode(),inplace=True)
In [15]:
data.elevator.fillna(data.elevator.mode(),inplace=True)
In [16]:
data.fiveYearsProperty.fillna(data.fiveYearsProperty.mode(),inplace=True)
In [17]:
data.subway.fillna(data.subway.median(),inplace=True)
In [18]:
data.communityAverage.fillna(data.communityAverage.median(),inplace=True)
In [19]:
data.livingRoom.unique()

data.floor.unique()
#so,floor have a chinese character...

data.bathRoom.unique()

data.bathRoom.unique()

def Trade_Time(x):
    return x[0:4]
data['tradeTime']=data['tradeTime'].apply(Trade_Time)
data.head()

#convert tradetime into int numeric
data['tradeTime'] = pd.to_numeric(data['tradeTime'])
data['livingRoom'] = data['livingRoom'].apply(pd.to_numeric, errors='coerce')
data['drawingRoom'] = data['drawingRoom'].apply(pd.to_numeric, errors='coerce')
data['bathRoom'] = data['bathRoom'].apply(pd.to_numeric, errors='coerce')
#convert ConstructionTime into int numeric
data['constructionTime'] = data['constructionTime'].apply(pd.to_numeric, errors='coerce')

#now if we check livingRoom Column it is clean data.
data.livingRoom.unique()

#Now,Split the column into a Floor_Type and Floor_Height
def Floor_Type(x):
    return x.split(' ')[0]

def Floor_Height(y):
    try:
        return int(y.split(' ')[1])
    except:
        return np.nan

data['floor_type']=data['floor'].apply(Floor_Type)    
data['floor_height']=data['floor'].apply(Floor_Height)
In [27]:
data.columns

data=data.drop(columns=['floor','url','id','Cid','price'])
data.head()

#Let's Perform one hot encoding
print(data.buildingType.unique())
print(data.renovationCondition.unique())
print(data.buildingStructure.unique())
#so,for buildingType we have a data like 0.5   0.333 0.125 0.25  0.429 0.048 0.375 0.667
# Which is unnecessary so,we have to remove them

#Removing unnecessary data which is present in buildingType
data=data[data['buildingType']>=1]
In [31]:
print(data.buildingType.unique())
print(data.shape)

#let's take a copy of our data for future use
df=data.copy()
In [33]:
col_for_dummies=['renovationCondition','buildingStructure','buildingType',
                 'district','elevator','floor_type']
data=pd.get_dummies(data=data,columns=col_for_dummies,drop_first=True)
In [34]:
data.head()

print(data.shape)
print(df0.shape)

data=data.dropna(axis=0)
In [37]:
print(data.shape)

data.columns

df1=data[['Lng','Lat','tradeTime','totalPrice','followers','followers','livingRoom','drawingRoom','kitchen',
    'bathRoom','square','communityAverage','ladderRatio']]
In [40]:
plt.figure(figsize=(20,20))
sns.heatmap(df1.corr(),annot=True,cmap = "RdYlGn")
plt.show()

sns.kdeplot(data=data['totalPrice'],shade=True)

data['totalPrice'].describe()

df.head()

sns.scatterplot(x=df['followers'],y=df['communityAverage'],hue=df['elevator'])

# sns.swarmplot(x=df['renovationCondition'],
#               y=df['followers'])

sns.lineplot(data=df['communityAverage'])

data.head()

data.shape

data.to_csv("After_EDA.csv")
