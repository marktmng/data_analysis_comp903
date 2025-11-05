import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load data with composite scores
df = pd.read_csv('survey_data_with_scores.csv')

print("="*80)
print("CORRELATION ANALYSIS - CYBERSECURITY SURVEY")
print("="*80)

# Define question columns
knowledge_cols = [
    'I understand what phishing is.',
    'I can recognise a phishing email or message when I see one.',
    'I know what two-factor authentication (2FA) is and why it is used.',
    'I understand what malware is and how it functions.',
    'I am familiar with what a Virtual Private Network (VPN) does.',
    'I know best practices for creating strong and secure passwords.',
    'I understand the risks of using public Wi-Fi without additional protections.'
]

practice_cols = [
    'I use a unique password for each of my important online accounts.',
    'I regularly update the software and applications on my devices.',
    'I enable two-factor authentication (2FA) on accounts whenever possible.',
    'I avoid clicking links or opening attachments from unknown or suspicious emails.',
    'I review app permissions and only allow access to what is necessary.',
    'I back up important files regularly to a secure location.',
    'I use a VPN or other protections when accessing public Wi-Fi for sensitive tasks.'
]

attitude_cols = [
    'I feel confident in my ability to detect online scams and phishing attempts.',
    'I am concerned about the privacy and security of my personal information online.',
    'I believe cybersecurity knowledge is important for my academic and professional life.',
    'I believe my current online practices are sufficient to protect me from most cyber threats.',
    'I would attend a workshop on practical cybersecurity skills if my institution offered one.'
]

# ==============================================================================
# 1. COMPOSITE SCORES CORRELATION
# ==============================================================================
print("\n1. COMPOSITE SCORES CORRELATION MATRIX")
print("-"*80)

composite_cols = ['Knowledge_Score', 'Practice_Score', 'Attitude_Score', 'Overall_Cybersecurity_Score']
composite_corr = df[composite_cols].corr()
print(composite_corr.round(3))

print("\n1.1 Key Correlations:")
print(f"Knowledge ↔ Practice: r = {composite_corr.loc['Knowledge_Score', 'Practice_Score']:.3f}")
print(f"Knowledge ↔ Attitude: r = {composite_corr.loc['Knowledge_Score', 'Attitude_Score']:.3f}")
print(f"Practice ↔ Attitude: r = {composite_corr.loc['Practice_Score', 'Attitude_Score']:.3f}")

# Statistical significance
r_kp, p_kp = pearsonr(df['Knowledge_Score'], df['Practice_Score'])
r_ka, p_ka = pearsonr(df['Knowledge_Score'], df['Attitude_Score'])
r_pa, p_pa = pearsonr(df['Practice_Score'], df['Attitude_Score'])

print(f"\n1.2 Statistical Significance (Pearson's r):")
print(f"Knowledge ↔ Practice: r = {r_kp:.3f}, p = {p_kp:.4f} {'***' if p_kp < 0.001 else '**' if p_kp < 0.01 else '*' if p_kp < 0.05 else 'ns'}")
print(f"Knowledge ↔ Attitude: r = {r_ka:.3f}, p = {p_ka:.4f} {'***' if p_ka < 0.001 else '**' if p_ka < 0.01 else '*' if p_ka < 0.05 else 'ns'}")
print(f"Practice ↔ Attitude: r = {r_pa:.3f}, p = {p_pa:.4f} {'***' if p_pa < 0.001 else '**' if p_pa < 0.01 else '*' if p_pa < 0.05 else 'ns'}")

# ==============================================================================
# 2. KNOWLEDGE ITEMS CORRELATION
# ==============================================================================
print("\n" + "="*80)
print("2. KNOWLEDGE ITEMS CORRELATION")
print("="*80)

knowledge_corr = df[knowledge_cols].corr()
print(knowledge_corr.round(3))

# Find strongest correlations in knowledge items
knowledge_pairs = []
for i in range(len(knowledge_cols)):
    for j in range(i+1, len(knowledge_cols)):
        corr_val = knowledge_corr.iloc[i, j]
        knowledge_pairs.append((knowledge_cols[i][:30], knowledge_cols[j][:30], corr_val))

knowledge_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nTop 5 Strongest Correlations in Knowledge Items:")
for i, (col1, col2, corr) in enumerate(knowledge_pairs[:5], 1):
    print(f"{i}. r = {corr:.3f}: {col1}... ↔ {col2}...")

# ==============================================================================
# 3. PRACTICE ITEMS CORRELATION
# ==============================================================================
print("\n" + "="*80)
print("3. PRACTICE ITEMS CORRELATION")
print("="*80)

practice_corr = df[practice_cols].corr()
print(practice_corr.round(3))

# Find strongest correlations in practice items
practice_pairs = []
for i in range(len(practice_cols)):
    for j in range(i+1, len(practice_cols)):
        corr_val = practice_corr.iloc[i, j]
        practice_pairs.append((practice_cols[i][:30], practice_cols[j][:30], corr_val))

practice_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nTop 5 Strongest Correlations in Practice Items:")
for i, (col1, col2, corr) in enumerate(practice_pairs[:5], 1):
    print(f"{i}. r = {corr:.3f}: {col1}... ↔ {col2}...")

# ==============================================================================
# 4. ATTITUDE ITEMS CORRELATION
# ==============================================================================
print("\n" + "="*80)
print("4. ATTITUDE ITEMS CORRELATION")
print("="*80)

attitude_corr = df[attitude_cols].corr()
print(attitude_corr.round(3))

# Find strongest correlations in attitude items
attitude_pairs = []
for i in range(len(attitude_cols)):
    for j in range(i+1, len(attitude_cols)):
        corr_val = attitude_corr.iloc[i, j]
        attitude_pairs.append((attitude_cols[i][:30], attitude_cols[j][:30], corr_val))

attitude_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nTop 5 Strongest Correlations in Attitude Items:")
for i, (col1, col2, corr) in enumerate(attitude_pairs[:5], 1):
    print(f"{i}. r = {corr:.3f}: {col1}... ↔ {col2}...")

# ==============================================================================
# 5. CROSS-DOMAIN CORRELATIONS
# ==============================================================================
print("\n" + "="*80)
print("5. CROSS-DOMAIN CORRELATIONS")
print("="*80)

print("\n5.1 Knowledge Items vs Practice Score:")
for col in knowledge_cols:
    r, p = pearsonr(df[col].dropna(), df.loc[df[col].notna(), 'Practice_Score'])
    print(f"{col[:50]:<52} r = {r:.3f}, p = {p:.4f}")

print("\n5.2 Practice Items vs Knowledge Score:")
for col in practice_cols:
    r, p = pearsonr(df[col].dropna(), df.loc[df[col].notna(), 'Knowledge_Score'])
    print(f"{col[:50]:<52} r = {r:.3f}, p = {p:.4f}")

# ==============================================================================
# 6. ALL ITEMS CORRELATION OVERVIEW
# ==============================================================================
print("\n" + "="*80)
print("6. TOP 15 STRONGEST CORRELATIONS (ALL ITEMS)")
print("="*80)

all_items = knowledge_cols + practice_cols + attitude_cols
all_corr = df[all_items].corr()

# Extract all unique pairs
all_pairs = []
for i in range(len(all_items)):
    for j in range(i+1, len(all_items)):
        corr_val = all_corr.iloc[i, j]
        all_pairs.append((all_items[i][:40], all_items[j][:40], corr_val))

all_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

for i, (col1, col2, corr) in enumerate(all_pairs[:15], 1):
    print(f"{i:2d}. r = {corr:6.3f}: {col1}... ↔ {col2}...")

# ==============================================================================
# 7. VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("7. GENERATING CORRELATION VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Composite Scores Correlation Heatmap
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(composite_corr, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', ax=ax1, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1,
            square=True, linewidths=1)
ax1.set_title('Composite Scores Correlation', fontweight='bold', fontsize=12)

# 2. Knowledge Items Correlation
ax2 = fig.add_subplot(gs[0, 1])
sns.heatmap(knowledge_corr, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=ax2, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1,
            xticklabels=range(1, 8), yticklabels=range(1, 8), square=True)
ax2.set_title('Knowledge Items Correlation', fontweight='bold', fontsize=12)

# 3. Practice Items Correlation
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(practice_corr, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=ax3, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1,
            xticklabels=range(1, 8), yticklabels=range(1, 8), square=True)
ax3.set_title('Practice Items Correlation', fontweight='bold', fontsize=12)

# 4. Attitude Items Correlation
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(attitude_corr, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', ax=ax4, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1,
            xticklabels=range(1, 6), yticklabels=range(1, 6), square=True)
ax4.set_title('Attitude Items Correlation', fontweight='bold', fontsize=12)

# 5. Scatter: Knowledge vs Practice
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(df['Knowledge_Score'], df['Practice_Score'], alpha=0.6, s=100, color='steelblue')
z = np.polyfit(df['Knowledge_Score'], df['Practice_Score'], 1)
p = np.poly1d(z)
ax5.plot(df['Knowledge_Score'].sort_values(), p(df['Knowledge_Score'].sort_values()), 
         "r--", linewidth=2, label=f'r = {r_kp:.3f}')
ax5.set_xlabel('Knowledge Score', fontsize=11, fontweight='bold')
ax5.set_ylabel('Practice Score', fontsize=11, fontweight='bold')
ax5.set_title('Knowledge vs Practice', fontweight='bold', fontsize=12)
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Scatter: Knowledge vs Attitude
ax6 = fig.add_subplot(gs[1, 2])
ax6.scatter(df['Knowledge_Score'], df['Attitude_Score'], alpha=0.6, s=100, color='coral')
z = np.polyfit(df['Knowledge_Score'], df['Attitude_Score'], 1)
p = np.poly1d(z)
ax6.plot(df['Knowledge_Score'].sort_values(), p(df['Knowledge_Score'].sort_values()), 
         "r--", linewidth=2, label=f'r = {r_ka:.3f}')
ax6.set_xlabel('Knowledge Score', fontsize=11, fontweight='bold')
ax6.set_ylabel('Attitude Score', fontsize=11, fontweight='bold')
ax6.set_title('Knowledge vs Attitude', fontweight='bold', fontsize=12)
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Scatter: Practice vs Attitude
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(df['Practice_Score'], df['Attitude_Score'], alpha=0.6, s=100, color='lightgreen')
z = np.polyfit(df['Practice_Score'], df['Attitude_Score'], 1)
p = np.poly1d(z)
ax7.plot(df['Practice_Score'].sort_values(), p(df['Practice_Score'].sort_values()), 
         "r--", linewidth=2, label=f'r = {r_pa:.3f}')
ax7.set_xlabel('Practice Score', fontsize=11, fontweight='bold')
ax7.set_ylabel('Attitude Score', fontsize=11, fontweight='bold')
ax7.set_title('Practice vs Attitude', fontweight='bold', fontsize=12)
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Correlation Matrix - All Composite Scores (larger)
ax8 = fig.add_subplot(gs[2, 1:])
# Create a more detailed view
composite_data = df[composite_cols].corr()
mask = np.triu(np.ones_like(composite_data, dtype=bool), k=1)
sns.heatmap(composite_data, annot=True, cmap='RdYlGn', center=0,
            fmt='.3f', ax=ax8, cbar_kws={'shrink': 0.8}, vmin=-1, vmax=1,
            square=True, linewidths=2, linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'})
ax8.set_title('Composite Scores - Detailed Correlation Matrix', fontweight='bold', fontsize=13)

plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Correlation visualizations saved as 'correlation_analysis.png'")

# ==============================================================================
# 8. CORRELATION STRENGTH INTERPRETATION
# ==============================================================================
print("\n" + "="*80)
print("8. CORRELATION INTERPRETATION SUMMARY")
print("="*80)

def interpret_correlation(r):
    abs_r = abs(r)
    if abs_r < 0.3:
        return "Weak"
    elif abs_r < 0.5:
        return "Moderate"
    elif abs_r < 0.7:
        return "Strong"
    else:
        return "Very Strong"

print("\nComposite Scores Relationships:")
print(f"Knowledge ↔ Practice: {interpret_correlation(r_kp)} correlation (r = {r_kp:.3f})")
print(f"Knowledge ↔ Attitude: {interpret_correlation(r_ka)} correlation (r = {r_ka:.3f})")
print(f"Practice ↔ Attitude: {interpret_correlation(r_pa)} correlation (r = {r_pa:.3f})")

print("\n" + "="*80)
print("CORRELATION ANALYSIS COMPLETE")
print("="*80)