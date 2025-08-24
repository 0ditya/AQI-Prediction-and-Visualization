# et_model_train.py

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("merged_aqi_weather.csv")
df = df.drop(columns=['From Date', 'DATE', 'pm10_imputed'], errors='ignore')
df = df.dropna(subset=['PM2.5 (ug/m3)', 'PM10 (ug/m3)'])

# One-hot encode 'conditions' column if exists
if 'conditions' in df.columns:
    df = pd.get_dummies(df, columns=['conditions'], drop_first=True)


target = 'PM2.5 (ug/m3)'
X = df.drop(columns=['PM2.5 (ug/m3)', 'PM10 (ug/m3)'])
y = df[target]

imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Extra Trees
et = ExtraTreesRegressor(random_state=42)
et.fit(X_train, y_train)

# Predict and evaluate
y_pred = et.predict(X_test)
print("\n✅ Initial Extra Trees Performance")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Hyperparameter tuning
grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("\n🔍 Tuning Extra Trees Regressor...")
gs = GridSearchCV(estimator=et, param_grid=grid, cv=3, scoring='r2', n_jobs=-1)
gs.fit(X_train, y_train)

print("\n🏆 Best Params:", gs.best_params_)
print("✅ Best R2 Score:", gs.best_score_)

# Evaluate best model on test
y_best_pred = gs.best_estimator_.predict(X_test)
print("\n✅ Tuned Extra Trees Performance")
print(f"R2 Score: {r2_score(y_test, y_best_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_best_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_best_pred)):.2f}")

# Feature importance plot
importances = gs.best_estimator_.feature_importances_
features = X.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Extra Trees (PM2.5)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()