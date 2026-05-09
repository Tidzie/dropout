import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Loading data...")
df = pd.read_csv('zimbabwe_student_dropout.csv')

# Handle missing
df['family_income'] = df['family_income'].fillna(df['family_income'].median())
df['study_hours'] = df['study_hours'].fillna(df['study_hours'].median())
df['attendance_rate'] = df['attendance_rate'].fillna(df['attendance_rate'].median())
df['fees_paid'] = df['fees_paid'].fillna('Unknown')

# Encode
categorical_cols = ['gender', 'location', 'internet_access', 'electricity_reliability', 'previous_grade', 'fees_paid', 'part_time_job']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Feature engineering
df['academic_score'] = (df['study_hours'] * 0.3) + (df['attendance_rate'] * 0.4) + (df['previous_grade'] * 0.3)
df['engagement_score'] = df['lms_logins'] * df['study_hours']

df['academic_score'] = df['academic_score'].fillna(df['academic_score'].median())
df['engagement_score'] = df['engagement_score'].fillna(df['engagement_score'].median())

# Scaling
X = df.drop('dropout', axis=1)
y = df['dropout']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = np.nan_to_num(X_scaled)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Fit models
print("Training Logistic Regression...")
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Save
print("Saving models...")
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(dt_model, 'dt_model.joblib')
joblib.dump(rf_model, 'student_dropout_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Models re-trained and saved successfully.")
