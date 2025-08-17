# If you have iris.target which is an array of numeric labels (0, 1, 2)
df['flower_type'] = [iris.target_names[i] for i in iris.target]

# Or alternatively:
df['flower_type'] = pd.Series(iris.target).map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

df['target'] = iris.target
df['flower_type'] = [iris.target_names[i] for i in df['target']]

## Option 2
X_test['prediction'] = pred
final_df = X_test