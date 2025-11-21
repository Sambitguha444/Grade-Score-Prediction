# Grade-Score-Prediction
It uses RandomForest Model of Machine learning by using a Regressor from Supervised Learning finding the grade score of a student using his/her, Study hours and Previous Year scores.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from google.colab import files

print("Upload the PREVIOUS SCORE dataset (the one with math/reading/writing scores)")
prev_uploaded = files.upload()

print("Upload the STUDY HOURS dataset (Hours, Scores)")
hours_uploaded = files.upload()

prev_filename = list(prev_uploaded.keys())[0]
hours_filename = list(hours_uploaded.keys())[0]

prev_df = pd.read_csv(prev_filename)
hours_df = pd.read_csv(hours_filename)

print("\nPrevious Score Dataset Preview:")
display(prev_df.head())

print("\nStudy Hours Dataset Preview:")
display(hours_df.head())

prev_clean = prev_df[["math score", "reading score", "writing score"]].copy()

hours_clean = hours_df.rename(columns={"Hours": "study_hours"}).copy()

merged_df = pd.concat([prev_clean, hours_clean], axis=1)

merged_df["final_grade"] = merged_df[["math score", "reading score", "writing score"]].mean(axis=1)

print("\nMerged Dataset Preview:")
display(merged_df.head())

X = merged_df[["study_hours", "math score", "reading score", "writing score"]]
y = merged_df["final_grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nModel Performance:")
print("Mean Absolute Error:", mean_absolute_error(y_test, pred))
print("R² Score:", r2_score(y_test, pred))
sample_student = [[5, 70, 75, 73]]
prediction = model.predict(sample_student)

print(f"\nPredicted Final Grade: {prediction[0]:.2f}")
