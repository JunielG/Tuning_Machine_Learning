## Option 1: Use float formatting ##
sns.heatmap(overlap_matrix, annot=True, fmt=".2f", 
            xticklabels=list(selection_methods.keys()),
            yticklabels=list(selection_methods.keys()),
            cmap="YlGnBu")

## Option 2: Convert to integers first (if appropriate) ##
sns.heatmap(overlap_matrix.astype(int), annot=True, fmt="d", 
            xticklabels=list(selection_methods.keys()),
            yticklabels=list(selection_methods.keys()),
            cmap="YlGnBu")

## Option 3: Use general formatting ##
sns.heatmap(overlap_matrix, annot=True, fmt="g", 
            xticklabels=list(selection_methods.keys()),
            yticklabels=list(selection_methods.keys()),
            cmap="YlGnBu")

## Option 4: Let seaborn auto-format ##
sns.heatmap(overlap_matrix, annot=True,  # Remove fmt parameter
            xticklabels=list(selection_methods.keys()),
            yticklabels=list(selection_methods.keys()),
            cmap="YlGnBu")

The most common choice would be Option 1 with fmt=".2f" if you want to show 2 decimal places, or Option 2 if your overlap values should logically be integers (like counts).