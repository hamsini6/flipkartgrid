# Traffic Demand Prediction 

## 1. Problem Overview

The objective of this project is to predict traffic demand using historical transportation, road, location, and weather-related information. The target variable is **demand**, and model performance is evaluated using the **R² score**.

The dataset contains spatial, temporal, infrastructure, and environmental attributes that collectively influence traffic flow patterns. Therefore, the problem is treated as a supervised machine learning regression task.

---

## 2. Data Understanding

The dataset contains the following information:

### Location Information

* Geohash

### Temporal Information

* Day
* Timestamp

### Road Infrastructure Information

* RoadType
* NumberofLanes
* LargeVehicles
* Landmarks

### Environmental Information

* Temperature
* Weather

### Target Variable

* Demand

Traffic demand depends on multiple factors such as location, time of day, weather conditions, and road characteristics. Therefore, extracting meaningful information from these features is essential for accurate predictions.

---

## 3. Data Preprocessing

### Missing Value Handling

Some columns contained missing values which were handled before model training.

**RoadType**

* Missing values were replaced with "Missing_Road".

**Weather**

* Missing values were replaced with "Missing_Weather".

**Temperature**

* Missing values were imputed using the median temperature corresponding to the same day and timestamp.
* Any remaining missing values were replaced using the overall median temperature.

This approach preserves the temporal characteristics of temperature and avoids losing potentially useful records.

---

## 4. Geospatial Feature Engineering

### Geohash Decoding

The geohash feature contains encoded geographic location information.

Using the PyGeoHash library, each geohash was decoded into:

* Latitude
* Longitude

These numerical coordinates provide direct spatial information to the machine learning models.

### Geohash Prefix

A geohash prefix of length five was extracted.

Nearby locations often share common geohash prefixes. This helps the model learn regional traffic patterns.

### Frequency Encoding

Frequency encoding was applied to:

* Geohash
* Geohash Prefix

Each location was replaced with the number of times it appeared in the dataset. This helps identify frequently visited and less frequently visited areas.

---

## 5. Temporal Feature Engineering

Traffic demand is strongly dependent on time.

The timestamp feature was split into:

### Hour

Extracted the hour component from the timestamp.

Example:
08:15 → 8

### Minute

Extracted the minute component.

Example:
08:15 → 15

### Total Minutes

Converted the timestamp into minutes from midnight.

Example:
08:15 = 8 × 60 + 15 = 495

This provides a continuous numerical representation of time.

---

## 6. Cyclical Time Features

Time is cyclical in nature. For example, 23:59 and 00:00 are very close in reality but numerically far apart.

To capture this behavior, cyclical transformations were applied:

* Sine transformation
* Cosine transformation

These features allow machine learning models to understand repeating daily traffic patterns more effectively.

---

## 7. Rush Hour Detection

Traffic congestion usually increases during peak commuting periods.

A binary feature called **is_rush_hour** was created.

Rush hours were defined as:

* Morning: 8 AM to 10 AM
* Evening: 5 PM to 7 PM

This feature helps capture periods of consistently high traffic demand.

---

## 8. Interaction Features

Additional interaction features were created to model relationships between variables.

### Lane-Vehicle Interaction

A combined feature was created using:

* NumberofLanes
* LargeVehicles

This captures how road capacity interacts with heavy vehicle permissions.

### Temperature-Weather Interaction

A combined feature was created using:

* Weather
* Temperature

Traffic conditions often depend on the combined effect of temperature and weather rather than either variable alone.

---

## 9. Categorical Encoding

Machine learning algorithms require numerical inputs.

The following categorical variables were encoded:

* RoadType
* LargeVehicles
* Landmarks
* Weather
* Geohash
* Geohash Prefix
* Lane-Vehicle Interaction
* Temperature-Weather Interaction

Label Encoding was applied to convert these categories into numerical values.

---

## 10. Target Encoding

A target encoding strategy was applied to the geohash feature.

The average demand for each location was calculated and used as an additional feature.

Since target encoding can introduce data leakage, an out-of-fold approach was implemented using cross-validation.

For each validation fold:

* Encoding values were calculated only from the training folds.
* Validation data was never used when computing its own encoding values.

This ensured a leakage-safe implementation.

---

## 11. Cross Validation Strategy

A 5-Fold K-Fold Cross Validation approach was used.

Benefits include:

* More reliable model evaluation.
* Reduced overfitting.
* Better estimation of real-world performance.

Each fold acts as a validation set once while the remaining folds are used for training.

---

## 12. Model Training

Three gradient boosting algorithms were trained and evaluated.

### LightGBM

LightGBM is efficient for large tabular datasets and can learn complex feature interactions quickly.

### XGBoost

XGBoost is a powerful boosting algorithm known for its strong predictive performance and robustness.

### CatBoost

CatBoost handles categorical information effectively and often performs exceptionally well on structured datasets.

Using multiple models helps capture different patterns present in the data.

---

## 13. Ensemble Learning

Instead of relying on a single model, predictions from all three models were combined using weighted averaging.

Weights used:

* LightGBM: 40%
* XGBoost: 35%
* CatBoost: 25%

Ensembling improves prediction stability and reduces model-specific errors.

---

## 14. Final Prediction Generation

After cross-validation, all models were retrained on the complete training dataset.

Predictions were generated for the test dataset and combined using the ensemble weights.

The final output was stored in a CSV file named **submission.csv** containing:

Index,demand

This file was submitted for evaluation.

---

## 15. Tools and Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost
* PyGeoHash

