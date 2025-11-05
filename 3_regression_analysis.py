import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data
df = pd.read_csv('survey_data_with_scores.csv')

print("="*80)
print("REGRESSION ANALYSIS - CYBERSECURITY SURVEY")
print("="*80)

# Encode training variable
df['Training_Binary'] = df['I have received formal cybersecurity training (e.g., course, workshop, or module) during my studies.'].map({'Yes': 1, 'No': 0})
df['Training_Binary'].fillna(0, inplace=True)

# ==============================================================================
# MODEL 1: Practice Score ~ Knowledge Score
# ==============================================================================
print("\n" + "="*80)
print("MODEL 1: SIMPLE LINEAR REGRESSION")
print("Dependent Variable: Practice Score")
print("Independent Variable: Knowledge Score")
print("="*80)

X1 = df[['Knowledge_Score']].values
y1 = df['Practice_Score'].values

model1 = LinearRegression()
model1.fit(X1, y1)
y1_pred = model1.predict(X1)

# Calculate metrics
r2_1 = r2_score(y1, y1_pred)
mse_1 = mean_squared_error(y1, y1_pred)
rmse_1 = np.sqrt(mse_1)
mae_1 = mean_absolute_error(y1, y1_pred)

print("\n1.1 Model Coefficients:")
print(f"Intercept (β₀): {model1.intercept_:.4f}")
print(f"Knowledge Score Coefficient (β₁): {model1.coef_[0]:.4f}")

print("\n1.2 Model Equation:")
print(f"Practice Score = {model1.intercept_:.4f} + {model1.coef_[0]:.4f} × Knowledge Score")

print("\n1.3 Model Performance:")
print(f"R-squared (R²): {r2_1:.4f}")
print(f"Adjusted R²: {1 - (1 - r2_1) * (len(y1) - 1) / (len(y1) - 2):.4f}")
print(f"RMSE: {rmse_1:.4f}")
print(f"MAE: {mae_1:.4f}")

print("\n1.4 Interpretation:")
print(f"• {r2_1*100:.2f}% of the variance in Practice Score is explained by Knowledge Score")
print(f"• For every 1-point increase in Knowledge Score, Practice Score increases by {model1.coef_[0]:.4f} points")
print(f"• The average prediction error is {mae_1:.4f} points")

# Statistical significance
n = len(y1)
k = 1  # number of predictors
f_stat = (r2_1 / k) / ((1 - r2_1) / (n - k - 1))
p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
print(f"\n1.5 Statistical Significance:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Result: {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at α = 0.05")

# ==============================================================================
# MODEL 2: Overall Score ~ Knowledge + Attitude
# ==============================================================================
print("\n" + "="*80)
print("MODEL 2: MULTIPLE LINEAR REGRESSION")
print("Dependent Variable: Overall Cybersecurity Score")
print("Independent Variables: Knowledge Score, Attitude Score")
print("="*80)

X2 = df[['Knowledge_Score', 'Attitude_Score']].values
y2 = df['Overall_Cybersecurity_Score'].values

model2 = LinearRegression()
model2.fit(X2, y2)
y2_pred = model2.predict(X2)

# Calculate metrics
r2_2 = r2_score(y2, y2_pred)
mse_2 = mean_squared_error(y2, y2_pred)
rmse_2 = np.sqrt(mse_2)
mae_2 = mean_absolute_error(y2, y2_pred)

print("\n2.1 Model Coefficients:")
print(f"Intercept (β₀): {model2.intercept_:.4f}")
print(f"Knowledge Score Coefficient (β₁): {model2.coef_[0]:.4f}")
print(f"Attitude Score Coefficient (β₂): {model2.coef_[1]:.4f}")

print("\n2.2 Model Equation:")
print(f"Overall Score = {model2.intercept_:.4f} + {model2.coef_[0]:.4f} × Knowledge + {model2.coef_[1]:.4f} × Attitude")

print("\n2.3 Model Performance:")
print(f"R-squared (R²): {r2_2:.4f}")
print(f"Adjusted R²: {1 - (1 - r2_2) * (len(y2) - 1) / (len(y2) - 3):.4f}")
print(f"RMSE: {rmse_2:.4f}")
print(f"MAE: {mae_2:.4f}")

print("\n2.4 Interpretation:")
print(f"• {r2_2*100:.2f}% of the variance in Overall Score is explained by Knowledge and Attitude")
print(f"• Holding Attitude constant, 1-point increase in Knowledge increases Overall Score by {model2.coef_[0]:.4f}")
print(f"• Holding Knowledge constant, 1-point increase in Attitude increases Overall Score by {model2.coef_[1]:.4f}")

# Statistical significance
n = len(y2)
k = 2
f_stat = (r2_2 / k) / ((1 - r2_2) / (n - k - 1))
p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
print(f"\n2.5 Statistical Significance:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Result: {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at α = 0.05")

# ==============================================================================
# MODEL 3: Practice Score ~ Knowledge + Attitude + Training
# ==============================================================================
print("\n" + "="*80)
print("MODEL 3: MULTIPLE LINEAR REGRESSION WITH TRAINING")
print("Dependent Variable: Practice Score")
print("Independent Variables: Knowledge Score, Attitude Score, Training Status")
print("="*80)

X3 = df[['Knowledge_Score', 'Attitude_Score', 'Training_Binary']].values
y3 = df['Practice_Score'].values

# Remove rows with NaN
mask = ~np.isnan(X3).any(axis=1) & ~np.isnan(y3)
X3_clean = X3[mask]
y3_clean = y3[mask]

model3 = LinearRegression()
model3.fit(X3_clean, y3_clean)
y3_pred = model3.predict(X3_clean)

# Calculate metrics
r2_3 = r2_score(y3_clean, y3_pred)
mse_3 = mean_squared_error(y3_clean, y3_pred)
rmse_3 = np.sqrt(mse_3)
mae_3 = mean_absolute_error(y3_clean, y3_pred)

print("\n3.1 Model Coefficients:")
print(f"Intercept (β₀): {model3.intercept_:.4f}")
print(f"Knowledge Score Coefficient (β₁): {model3.coef_[0]:.4f}")
print(f"Attitude Score Coefficient (β₂): {model3.coef_[1]:.4f}")
print(f"Training Status Coefficient (β₃): {model3.coef_[2]:.4f}")

print("\n3.2 Model Equation:")
print(f"Practice Score = {model3.intercept_:.4f} + {model3.coef_[0]:.4f} × Knowledge + {model3.coef_[1]:.4f} × Attitude + {model3.coef_[2]:.4f} × Training")

print("\n3.3 Model Performance:")
print(f"R-squared (R²): {r2_3:.4f}")
print(f"Adjusted R²: {1 - (1 - r2_3) * (len(y3_clean) - 1) / (len(y3_clean) - 4):.4f}")
print(f"RMSE: {rmse_3:.4f}")
print(f"MAE: {mae_3:.4f}")

print("\n3.4 Interpretation:")
print(f"• {r2_3*100:.2f}% of variance in Practice Score is explained by Knowledge, Attitude, and Training")
print(f"• Holding other variables constant:")
print(f"  - 1-point increase in Knowledge increases Practice Score by {model3.coef_[0]:.4f}")
print(f"  - 1-point increase in Attitude increases Practice Score by {model3.coef_[1]:.4f}")
print(f"  - Having formal training changes Practice Score by {model3.coef_[2]:.4f} points")

# Statistical significance
n = len(y3_clean)
k = 3
f_stat = (r2_3 / k) / ((1 - r2_3) / (n - k - 1))
p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)
print(f"\n3.5 Statistical Significance:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Result: {'Statistically significant' if p_value < 0.05 else 'Not statistically significant'} at α = 0.05")

# ==============================================================================
# MODEL COMPARISON
# ==============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_data = {
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'Dependent Variable': ['Practice', 'Overall', 'Practice'],
    'Predictors': ['Knowledge', 'Knowledge + Attitude', 'Knowledge + Attitude + Training'],
    'R²': [r2_1, r2_2, r2_3],
    'RMSE': [rmse_1, rmse_2, rmse_3],
    'MAE': [mae_1, mae_2, mae_3]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# ==============================================================================
# VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("GENERATING REGRESSION VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Model 1: Scatter with regression line
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X1, y1, alpha=0.6, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
ax1.plot(X1, y1_pred, 'r-', linewidth=3, label=f'R² = {r2_1:.3f}')
ax1.set_xlabel('Knowledge Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Practice Score', fontsize=12, fontweight='bold')
ax1.set_title('Model 1: Practice ~ Knowledge', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 2. Model 1: Residual plot
ax2 = fig.add_subplot(gs[0, 1])
residuals1 = y1 - y1_pred
ax2.scatter(y1_pred, residuals1, alpha=0.6, s=100, color='steelblue', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Practice Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax2.set_title('Model 1: Residual Plot', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Model 1: Q-Q plot
ax3 = fig.add_subplot(gs[0, 2])
stats.probplot(residuals1, dist="norm", plot=ax3)
ax3.set_title('Model 1: Q-Q Plot', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Model 2: Actual vs Predicted
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y2, y2_pred, alpha=0.6, s=100, color='coral', edgecolors='black', linewidth=0.5)
ax4.plot([y2.min(), y2.max()], [y2.min(), y2.max()], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Overall Score', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted Overall Score', fontsize=12, fontweight='bold')
ax4.set_title(f'Model 2: Actual vs Predicted (R² = {r2_2:.3f})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# 5. Model 2: Residual plot
ax5 = fig.add_subplot(gs[1, 1])
residuals2 = y2 - y2_pred
ax5.scatter(y2_pred, residuals2, alpha=0.6, s=100, color='coral', edgecolors='black', linewidth=0.5)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Overall Score', fontsize=12, fontweight='bold')
ax5.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax5.set_title('Model 2: Residual Plot', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Model 2: Coefficient plot
ax6 = fig.add_subplot(gs[1, 2])
coef_names_2 = ['Knowledge', 'Attitude']
coef_values_2 = model2.coef_
colors_2 = ['steelblue', 'coral']
bars = ax6.barh(coef_names_2, coef_values_2, color=colors_2, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax6.set_title('Model 2: Regression Coefficients', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, coef_values_2)):
    ax6.text(val, i, f' {val:.3f}', va='center', fontsize=11, fontweight='bold')

# 7. Model 3: Actual vs Predicted
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(y3_clean, y3_pred, alpha=0.6, s=100, color='lightgreen', edgecolors='black', linewidth=0.5)
ax7.plot([y3_clean.min(), y3_clean.max()], [y3_clean.min(), y3_clean.max()], 'r--', linewidth=2)
ax7.set_xlabel('Actual Practice Score', fontsize=12, fontweight='bold')
ax7.set_ylabel('Predicted Practice Score', fontsize=12, fontweight='bold')
ax7.set_title(f'Model 3: Actual vs Predicted (R² = {r2_3:.3f})', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Model 3: Coefficient plot
ax8 = fig.add_subplot(gs[2, 1])
coef_names_3 = ['Knowledge', 'Attitude', 'Training']
coef_values_3 = model3.coef_
colors_3 = ['steelblue', 'coral', 'lightgreen']
bars = ax8.barh(coef_names_3, coef_values_3, color=colors_3, edgecolor='black', linewidth=1.5)
ax8.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
ax8.set_title('Model 3: Regression Coefficients', fontsize=13, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, coef_values_3)):
    ax8.text(val, i, f' {val:.3f}', va='center', fontsize=11, fontweight='bold')

# 9. R² Comparison across models
ax9 = fig.add_subplot(gs[2, 2])
models = ['Model 1', 'Model 2', 'Model 3']
r2_values = [r2_1, r2_2, r2_3]
bars = ax9.bar(models, r2_values, color=['steelblue', 'coral', 'lightgreen'], 
               edgecolor='black', linewidth=1.5)
ax9.set_ylabel('R² Value', fontsize=12, fontweight='bold')
ax9.set_ylim(0, 1)
ax9.set_title('Model Comparison: R² Values', fontsize=13, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, r2_values):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.savefig('regression_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Regression visualizations saved as 'regression_analysis.png'")

print("\n" + "="*80)
print("REGRESSION ANALYSIS COMPLETE")
print("="*80)