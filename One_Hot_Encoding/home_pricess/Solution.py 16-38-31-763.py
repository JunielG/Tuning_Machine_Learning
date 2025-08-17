## Solution 1
df1['town'] = np.where(df1['town'] == 'monroe township', 0, df1['town'])
df1['town'] = np.where(df1['town'] == 'west windsor', 2, df1['town'])
df1['town'] = np.where(df1['town'] == 'robinsville', 1, df1['town'])

## Solution 2 
df['town'] = np.where(df['town'] == 'monroe township', 1, 0)

## Solution 3 
df['town'] = np.where(df['town'] == 'monroe township', df['town'], 0)

## Solution 4 
le = LabelEncoder()
df1.town = le.fit_transform(df1.town) #object & categorical col
