import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression

# Since you didn't provide actual data, I'll create some realistic sample data
# with expected correlations (newer cars with lower mileage tend to have higher prices)
np.random.seed(42)

# Generate sample data
n_samples = 100
age_years = np.random.uniform(1, 15, n_samples)
mileage_base = age_years * 10000  # Base mileage proportional to age
mileage = mileage_base + np.random.normal(0, 5000, n_samples)  # Add some noise
# Sell price negatively correlated with both age and mileage
sell_price = 30000 - 1500 * age_years - 0.1 * mileage / 1000 + np.random.normal(0, 2000, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'age_years': age_years,
    'mileage': mileage,
    'sell_price': sell_price
})

# Make sure all values make sense (no negative prices or mileage)
data['sell_price'] = np.maximum(data['sell_price'], 1000)
data['mileage'] = np.maximum(data['mileage'], 0)


### MOST OF THE TIME I'LL USE THIS DOWN BELOW ###

# Create correlation plots
plt.figure(figsize=(16, 12))

# 1. Correlation matrix heatmap
plt.subplot(2, 2, 1)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix', fontsize=14)

# 2. Scatter plot: Age vs. Price
plt.subplot(2, 2, 2)
sns.scatterplot(x='age_years', y='sell_price', data=data)
plt.title('Age vs. Sell Price', fontsize=14)
plt.xlabel('Age (years)')
plt.ylabel('Sell Price ($)')

# 3. Scatter plot: Mileage vs. Price
plt.subplot(2, 2, 3)
sns.scatterplot(x='mileage', y='sell_price', data=data)
plt.title('Mileage vs. Sell Price', fontsize=14)
plt.xlabel('Mileage (miles)')
plt.ylabel('Sell Price ($)')

# 4. Scatter plot: Age vs. Mileage
plt.subplot(2, 2, 4)
sns.scatterplot(x='age_years', y='mileage', data=data)
plt.title('Age vs. Mileage', fontsize=14)
plt.xlabel('Age (years)')
plt.ylabel('Mileage (miles)')

# Add a main title
plt.suptitle('Correlation Analysis of Vehicle Age, Mileage, and Sell Price', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show plot
plt.show()

# Optional: 3D scatter plot to visualize all three variables together
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['age_years'], data['mileage'], data['sell_price'])
ax.set_xlabel('Age (years)')
ax.set_ylabel('Mileage (miles)')
ax.set_zlabel('Sell Price ($)')
ax.set_title('3D Relationship: Age, Mileage, and Sell Price')
plt.tight_layout()
plt.show()

# Print correlation values
print("Correlation coefficients:")
print(correlation_matrix)