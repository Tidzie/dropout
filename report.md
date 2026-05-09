# PROJECT REPORT: PREDICTING STUDENT DROPOUT USING MACHINE LEARNING

**Institution:** Great Zimbabwe University  
**Department:** Mathematics and Computer Science  
**Course Code:** HCS221  
**Date:** 16 April 2026  

---

## 1. Executive Summary
This project aims to address the challenge of student attrition at Great Zimbabwe University by leveraging machine learning. By analyzing socio-economic and academic data from over 500,000 students, we developed a predictive model that identifies students at high risk of dropping out. The final model achieves high ROC-AUC performance, providing a reliable tool for early intervention.

## 2. Methodology
The project followed a standard machine learning pipeline:

### 2.1 Data Acquisition & Exploratory Data Analysis (EDA)
The dataset includes 14 features ranging from demographic data (age, gender, location) to financial status (family income, fees paid) and academic performance (attendance, study hours, previous grades: 1, 2.1, 2.2, 3, and Fail).
- **Key Finding:** There is a strong negative correlation between attendance rate and dropout status.
- **Key Finding:** Financial factors, specifically `fees_paid` and `family_income`, are significant predictors of student stability.

### 2.2 Data Pre-processing
Data was cleaned using the following techniques:
- **Imputation:** Median values were used for missing numerical data to maintain distribution integrity. "Unknown" was used for missing categorical financial data.
- **Encoding:** Categorical variables were transformed using `LabelEncoder`.
- **Scaling:** Features were normalized using `StandardScaler` to ensure that models like Logistic Regression treat all features with equal weight.

### 2.3 Feature Engineering
Two composite features were created to enhance predictive power:
1. **Academic Score:** A weighted combination of study hours, attendance, and previous grades.
2. **Engagement Score:** A product of LMS logins and study hours, highlighting students' active participation.

## 3. Model Development & Comparison
Three algorithms were evaluated:
1. **Logistic Regression:** Used as a baseline model.
2. **Decision Tree:** Captured non-linear relationships but showed signs of overfitting.
3. **Random Forest:** Selected as the final model due to its robust performance and ability to handle class imbalance using the `class_weight='balanced'` parameter. Hyperparameter tuning was performed via `GridSearchCV`.

## 4. Results & Discussion
### 4.1 Evaluation Metrics
The Random Forest model outperformed other models across all metrics:
- **Accuracy:** >90% (approx)
- **ROC-AUC:** High (indicating excellent separation between classes)
- **Recall:** Optimized to ensure that "at-risk" students are not missed.

### 4.2 Socio-Economic Impact
The analysis confirms that student success is not solely dependent on academic ability. Socio-economic factors such as location (Rural vs Urban) and financial reliability play a massive role. Students from rural areas with poor internet access and electricity are at a higher risk, suggesting a need for better infrastructure support.

## 5. Deployment & Integration
The model was saved using `joblib` and integrated into a **Streamlit** dashboard. This application allows university administrators to:
- Input student data manually.
- Receive a real-time risk assessment.
- View key engagement and academic metrics for the student.

## 6. Recommendations
1. **Financial Aid:** Target students identified as "High Risk" with financial counseling or scholarship opportunities.
2. **Infrastructure:** Improve internet and electricity access on campus to support rural and disadvantaged students.
3. **Monitoring:** Use the Engagement Score to flag students who stop logging into the LMS early in the semester.

---

## Appendix: Implementation Details
The full implementation can be found in the attached Jupyter Notebook (`dropout_prediction.ipynb`) and the web application (`app.py`).
