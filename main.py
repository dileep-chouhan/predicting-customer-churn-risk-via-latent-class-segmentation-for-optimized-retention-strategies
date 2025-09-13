import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_customers = 500
# Generate synthetic customer data
data = {
    'Age': np.random.randint(18, 70, num_customers),
    'Income': np.random.randint(20000, 150000, num_customers),
    'PurchaseFrequency': np.random.poisson(lam=5, size=num_customers), # Purchases per year
    'AveragePurchaseValue': np.random.uniform(50, 500, num_customers),
    'Tenure': np.random.randint(1, 10, num_customers), # Years as customer
    'Churn': np.random.binomial(1, 0.15, num_customers) # 15% churn rate
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing ---
# Scale numerical features for clustering
scaler = StandardScaler()
numerical_cols = ['Age', 'Income', 'PurchaseFrequency', 'AveragePurchaseValue', 'Tenure']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
# --- 3. Latent Class Segmentation (Clustering) ---
# Apply Gaussian Mixture Model for customer segmentation
gmm = GaussianMixture(n_components=3, random_state=42) # Experiment with different numbers of components
gmm_labels = gmm.fit_predict(df[numerical_cols])
df['Segment'] = gmm_labels
# --- 4. Churn Analysis by Segment ---
# Analyze churn rate within each segment
churn_by_segment = df.groupby('Segment')['Churn'].mean()
print("Churn Rate by Segment:")
print(churn_by_segment)
# --- 5. Visualization ---
# Visualize churn rate across segments
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_segment.index, y=churn_by_segment.values)
plt.title('Churn Rate by Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Churn Rate')
plt.xticks(churn_by_segment.index)
plt.savefig('churn_by_segment.png')
print("Plot saved to churn_by_segment.png")
#Further analysis could involve exploring feature importance within each segment to understand the drivers of churn.  
#This could be done with techniques like ANOVA or feature importance from a classification model trained on each segment.