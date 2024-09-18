**Wildfire Prediction anad Heatmap visualization**
The goal of this project is to predict the likelihood of wildfires based on the satellite features provided and generate a heatmap to visualize potential wildfire locations. 
This project makes use of the Random Forest Classifier to build the prediction model and Folium to create a geographical heatmap of wildfire predictions.

Data
The dataset used in this project is sourced from MODIS satellite data in South Africa for the year 2023. 
The dataset includes information such as latitude, longitude, brightness, scan, track, and fire radiative power (FRP).

Key Features
latitude: Geographic coordinate
longitude: Geographic coordinate
brightness: Brightness of the fire as captured by the satellite
scan and track: Satellite tracking data
bright_t31: Temperature at band 31 of the satellite
frp: Fire Radiative Power, the energy emitted by the fire

To run this project, you will need the following Python libraries:
pandas
numpy
scikit-learn
seaborn
matplotlib
folium

Model Metrics:
Accuracy: Evaluates the overall performance of the model.
Classification Report: Precision, recall, and F1-score for both classes.
Confusion Matrix: Breakdown of true positives, false positives, true negatives, and false negatives.

Results
Accuracy: The accuracy of the model on the test data.
Heatmap: The generated heatmap shows regions where wildfires are predicted to occur based on the satellite data.
