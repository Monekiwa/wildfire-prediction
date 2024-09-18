import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

data = pd.read_csv('modis_2023_South_Africa.csv')

data['wildfire'] = np.where(data['confidence'] > 80, 1, 0)

features = ['latitude', 'longitude', 'brightness', 'scan', 'track', 'bright_t31', 'frp']
X = data[features]
y = data['wildfire']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance for Wildfire Detection')
plt.show()

X_test['prediction'] = y_pred

wildfire_predictions = X_test[X_test['prediction'] == 1]

wildfire_map = folium.Map(location=[wildfire_predictions['latitude'].mean(), wildfire_predictions['longitude'].mean()], zoom_start=6)

heat_data = [[row['latitude'], row['longitude'], row['brightness']] for index, row in wildfire_predictions.iterrows()]

HeatMap(heat_data, radius=10).add_to(wildfire_map)

wildfire_map.save("wildfire_heatmap.html")

wildfire_map
