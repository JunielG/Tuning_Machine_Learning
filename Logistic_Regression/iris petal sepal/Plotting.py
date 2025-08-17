## If you want to plot predictions against a single feature:

_, ax = plt.subplots()
scatter = ax.scatter(pred, X_test[:, 0], c=iris.target)
ax.set(xlabel='Predicted', ylabel=iris.feature_names[0])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)

## Or if you want to compare two specific features:

_, ax = plt.subplots()
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
