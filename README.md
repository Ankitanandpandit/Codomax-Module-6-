Codomaxdigitalsolutions - Module 6 Task Completed 
Codomax Module 6 Task Completed Data Science Final Project 


Module 6: Final Data Science Project
Day 21 – Day 24 | High Priority
=============================================================
End-to-End Data Science Project
• Full analysis pipeline
• Feature engineering
• Machine Learning models
• Evaluation &amp; insights
• Portfolio-ready structure
Project: Customer Churn Prediction &amp; Business Insights
(Synthetic but realistic telecom-style dataset)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
 confusion_matrix, roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
print("■ Libraries imported successfully!")
# ============================================================
# 1. CREATE / LOAD REALISTIC DATASET
# ============================================================
print("\n" + "="*60)
print("■ STEP 1: DATASET CREATION (Realistic Customer Churn)")
print("="*60)
np.random.seed(42)
n = 3000
# Generate realistic features
tenure = np.random.randint(1, 73, n)
monthly_charges = np.random.uniform(20, 120, n)
total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
internet = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20])
payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)
senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice([0, 1], n, p=[0.48, 0.52])
dependents = np.random.choice([0, 1], n, p=[0.70, 0.30])
tech_support = np.random.choice(['Yes', 'No', 'No internet'], n, p=[0.30, 0.50, 0.20])
paperless = np.random.choice([0, 1], n, p=[0.40, 0.60])
# Create churn probability based on realistic rules
churn_prob = (
 0.15 +
 (contract == 'Month-to-month') * 0.25 +
 (internet == 'Fiber optic') * 0.12 +
 (tenure &lt; 12) * 0.18 +
 (monthly_charges &gt; 80) * 0.10 +
 (tech_support == 'No') * 0.08 +
 (senior == 1) * 0.05 -
 (contract == 'Two year') * 0.20 -
 (partner == 1) * 0.05
)
churn_prob = np.clip(churn_prob, 0.05, 0.85)
churn = (np.random.random(n) &lt; churn_prob).astype(int)
df = pd.DataFrame({
 'tenure': tenure,
 'MonthlyCharges': monthly_charges.round(2),
 'TotalCharges': total_charges.round(2),
 'Contract': contract,
 'InternetService': internet,
 'PaymentMethod': payment,
 'SeniorCitizen': senior,
 'Partner': partner,
 'Dependents': dependents,
 'TechSupport': tech_support,
 'PaperlessBilling': paperless,
 'Churn': churn
})
print(f"Dataset shape: {df.shape}")
print(f"Churn Rate: {df['Churn'].mean()*100:.1f}%")
print("\nFirst 5 rows:")
print(df.head())
# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n" + "="*60)
print("■ STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)
print("\n--- Target Distribution ---")
print(df['Churn'].value_counts(normalize=True).round(3) * 100)
print("\n--- Churn Rate by Contract Type ---")

print(df.groupby('Contract')['Churn'].mean().round(3) * 100)
print("\n--- Churn Rate by Internet Service ---")
print(df.groupby('InternetService')['Churn'].mean().round(3) * 100)
# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(data=df, x='Churn', ax=axes[0,0], palette=['#2ecc71', '#e74c3c'])
axes[0,0].set_title('Churn Distribution', fontweight='bold')
axes[0,0].set_xticklabels(['No Churn', 'Churned'])
sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[0,1], palette=['#2ecc71', '#e74c3c'])
axes[0,1].set_title('Tenure vs Churn', fontweight='bold')
axes[0,1].set_xticklabels(['No Churn', 'Churned'])
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0], palette=['#2ecc71', '#e74c3c'])
axes[1,0].set_title('Monthly Charges vs Churn', fontweight='bold')
axes[1,0].set_xticklabels(['No Churn', 'Churned'])
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
contract_churn.plot(kind='bar', ax=axes[1,1], color='#9b59b6')
axes[1,1].set_title('Churn Rate (%) by Contract', fontweight='bold')
axes[1,1].set_ylabel('Churn Rate %')
axes[1,1].tick_params(axis='x', rotation=15)
plt.tight_layout()
plt.show()
# ============================================================
# 3. DATA PREPROCESSING &amp; FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("■ STEP 3: PREPROCESSING &amp; FEATURE ENGINEERING")
print("="*60)
df_ml = df.copy()
# Encode categorical variables
le_cols = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']
for col in le_cols:
 le = LabelEncoder()
 df_ml[col] = le.fit_transform(df_ml[col])
# Feature Engineering
df_ml['AvgChargesPerMonth'] = df_ml['TotalCharges'] / (df_ml['tenure'] + 1)
df_ml['IsLongTerm'] = (df_ml['tenure'] &gt; 24).astype(int)
df_ml['HighSpender'] = (df_ml['MonthlyCharges'] &gt; df_ml['MonthlyCharges'].median()).astype(int)
print("■ Categorical encoding done")
print("■ New features created: AvgChargesPerMonth, IsLongTerm, HighSpender")
# Features &amp; Target
feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
 'InternetService', 'PaymentMethod', 'SeniorCitizen', 
 'Partner', 'Dependents', 'TechSupport', 'PaperlessBilling',
 'AvgChargesPerMonth', 'IsLongTerm', 'HighSpender']
X = df_ml[feature_cols]
y = df_ml['Churn']
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42, stratify=y
)
# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
# ============================================================
# 4. MODEL TRAINING &amp; EVALUATION
# ============================================================
print("\n" + "="*60)
print("■ STEP 4: MODEL TRAINING &amp; EVALUATION")
print("="*60)
models = {
 'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
 'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
 model.fit(X_train_scaled, y_train)
 y_pred = model.predict(X_test_scaled)
 y_prob = model.predict_proba(X_test_scaled)[:, 1]
 
 acc = accuracy_score(y_test, y_pred)
 auc = roc_auc_score(y_test, y_prob)
 
 results[name] = {'model': model, 'accuracy': acc, 'auc': auc, 
 'y_pred': y_pred, 'y_prob': y_prob}
 
 print(f"\n{name}:")
 print(f" Accuracy: {acc*100:.2f}%")
 print(f" ROC-AUC: {auc:.4f}")
# Best model
best_name = max(results, key=lambda x: results[x]['auc'])
best = results[best_name]
print(f"\n■ Best Model: {best_name} (AUC = {best['auc']:.4f})")
print("\n■ Classification Report (Best Model):")
print(classification_report(y_test, best['y_pred'], target_names=['No Churn', 'Churn']))
# ============================================================
# 5. VISUALIZE MODEL PERFORMANCE
# ============================================================
print("\n■ Creating evaluation plots...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
 xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title(f'Confusion Matrix - {best_name}', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
# ROC Curve
for name, res in results.items():
 fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
 axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[1].plot([0,1], [0,1], 'k--', alpha=0.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves Comparison', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print("\n■ Feature Importance (Random Forest):")
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
importances = importances.sort_values(ascending=True)
plt.figure(figsize=(9, 6))
importances.plot(kind='barh', color='#3498db')
plt.title('Feature Importance - Random Forest', fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
print(importances.sort_values(ascending=False).head(8))
# ============================================================
# 7. BUSINESS INSIGHTS &amp; RECOMMENDATIONS
# ============================================================
print("\n" + "="*60)
print("■ STEP 5: BUSINESS INSIGHTS &amp; RECOMMENDATIONS")
print("="*60)
print("""
■ KEY FINDINGS:
1. Contract Type is the strongest predictor of churn.
 → Month-to-month customers churn at much higher rates.
 → Long-term contracts (1-2 year) show strong retention.
2. Tenure matters significantly.
 → New customers (low tenure) are at highest risk.
 → After ~24 months, churn probability drops sharply.
3. Monthly Charges &amp; Internet Service influence churn.
 → High monthly charges + Fiber optic correlate with higher churn.
 → Possibly due to price sensitivity or service expectations.
4. Tech Support and Partner status act as retention factors.
■ ACTIONABLE RECOMMENDATIONS:
• Retention Campaign: Target month-to-month customers in first 12 months
 with personalized offers to upgrade to annual contracts.
• Pricing Review: Analyze high-charge Fiber customers for win-back
 discounts or value-added services (e.g., free Tech Support).
• Onboarding Excellence: Improve early-tenure experience — this is
 the highest risk window.
• Predictive Alerts: Deploy the best model to score customers weekly
 and trigger proactive outreach for high-risk profiles.
This end-to-end project demonstrates:
■ Data generation / loading
■ Thorough EDA &amp; visualization
■ Feature engineering
■ Multiple ML models + proper evaluation
■ Business storytelling from data
""")
print("\n■ FINAL DATA SCIENCE PROJECT COMPLETE!")
print("This is portfolio-ready. Upload to GitHub + share on LinkedIn.")
print("="*60)


Final Submission Checklist
■ Code runs without errors in Google Colab / locally
■ All plots and outputs are visible
■ Notebook / repo is shared publicly
■ LinkedIn post is live with correct links + screenshots
■ All required deliverable links submitted on the platform
This module is complete and portfolio-ready. Great work

