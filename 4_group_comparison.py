import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load data
df = pd.read_csv('survey_data_with_scores.csv')

print("="*80)
print("GROUP COMPARISON ANALYSIS - CYBERSECURITY SURVEY")
print("="*80)

# Encode training variable
df['Training_Binary'] = df['I have received formal cybersecurity training (e.g., course, workshop, or module) during my studies.'].map({'Yes': 1, 'No': 0})
df['Training_Binary'].fillna(0, inplace=True)

# ==============================================================================
# 1. COMPARISON BY TRAINING STATUS
# ==============================================================================
print("\n" + "="*80)
print("1. COMPARISON BY TRAINING STATUS")
print("="*80)

print("\n1.1 Descriptive Statistics by Training Status:")
training_stats = df.groupby('Training_Binary')[['Knowledge_Score', 'Practice_Score', 
                                                  'Attitude_Score', 'Overall_Cybersecurity_Score']].agg(['mean', 'std', 'count'])
print(training_stats)

# Independent t-tests for each score
print("\n1.2 Independent T-Tests (Training vs No Training):")

trained = df[df['Training_Binary'] == 1]
not_trained = df[df['Training_Binary'] == 0]

scores_to_test = ['Knowledge_Score', 'Practice_Score', 'Attitude_Score', 'Overall_Cybersecurity_Score']
score_names = ['Knowledge', 'Practice', 'Attitude', 'Overall']

for score, name in zip(scores_to_test, score_names):
    group1 = trained[score].dropna()
    group2 = not_trained[score].dropna()
    
    t_stat, p_val = stats.ttest_ind(group1, group2)
    cohen_d = (group1.mean() - group2.mean()) / np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
    
    print(f"\n{name} Score:")
    print(f"  Training: Mean = {group1.mean():.3f}, SD = {group1.std():.3f}, n = {len(group1)}")
    print(f"  No Training: Mean = {group2.mean():.3f}, SD = {group2.std():.3f}, n = {len(group2)}")
    print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
    print(f"  Cohen's d = {cohen_d:.3f} ({'Small' if abs(cohen_d) < 0.5 else 'Medium' if abs(cohen_d) < 0.8 else 'Large'} effect)")
    print(f"  Result: {'Significant difference' if p_val < 0.05 else 'No significant difference'} at α = 0.05")

# ==============================================================================
# 2. COMPARISON BY AGE GROUP
# ==============================================================================
print("\n" + "="*80)
print("2. COMPARISON BY AGE GROUP")
print("="*80)

print("\n2.1 Descriptive Statistics by Age Group:")
age_stats = df.groupby('Age group:')[['Knowledge_Score', 'Practice_Score', 
                                        'Attitude_Score', 'Overall_Cybersecurity_Score']].agg(['mean', 'std', 'count'])
print(age_stats)

# One-way ANOVA for each score
print("\n2.2 One-Way ANOVA (Age Groups):")

age_categories = df['Age group:'].dropna().unique()

for score, name in zip(scores_to_test, score_names):
    groups = [df[df['Age group:'] == age][score].dropna() for age in age_categories]
    groups = [g for g in groups if len(g) > 0]  # Remove empty groups
    
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        
        print(f"\n{name} Score:")
        print(f"  F-statistic = {f_stat:.3f}, p-value = {p_val:.4f}")
        print(f"  Result: {'Significant difference' if p_val < 0.05 else 'No significant difference'} between age groups at α = 0.05")
        
        # Post-hoc: Pairwise comparisons if significant
        if p_val < 0.05:
            print(f"  Post-hoc pairwise comparisons (with Bonferroni correction):")
            age_list = list(age_categories)
            for i in range(len(age_list)):
                for j in range(i+1, len(age_list)):
                    g1 = df[df['Age group:'] == age_list[i]][score].dropna()
                    g2 = df[df['Age group:'] == age_list[j]][score].dropna()
                    if len(g1) > 0 and len(g2) > 0:
                        t, p = stats.ttest_ind(g1, g2)
                        p_adj = p * len(age_list) * (len(age_list) - 1) / 2  # Bonferroni correction
                        if p_adj < 0.05:
                            print(f"    {age_list[i]} vs {age_list[j]}: p = {p_adj:.4f} *")

# ==============================================================================
# 3. COMPARISON BY GENDER
# ==============================================================================
print("\n" + "="*80)
print("3. COMPARISON BY GENDER")
print("="*80)

print("\n3.1 Descriptive Statistics by Gender:")
gender_stats = df.groupby('Gender:')[['Knowledge_Score', 'Practice_Score', 
                                       'Attitude_Score', 'Overall_Cybersecurity_Score']].agg(['mean', 'std', 'count'])
print(gender_stats)

# Filter to Male and Female only for t-test
gender_filtered = df[df['Gender:'].isin(['Male', 'Female'])]

print("\n3.2 Independent T-Tests (Male vs Female):")

for score, name in zip(scores_to_test, score_names):
    male = gender_filtered[gender_filtered['Gender:'] == 'Male'][score].dropna()
    female = gender_filtered[gender_filtered['Gender:'] == 'Female'][score].dropna()
    
    if len(male) > 0 and len(female) > 0:
        t_stat, p_val = stats.ttest_ind(male, female)
        cohen_d = (male.mean() - female.mean()) / np.sqrt(((len(male)-1)*male.std()**2 + (len(female)-1)*female.std()**2) / (len(male)+len(female)-2))
        
        print(f"\n{name} Score:")
        print(f"  Male: Mean = {male.mean():.3f}, SD = {male.std():.3f}, n = {len(male)}")
        print(f"  Female: Mean = {female.mean():.3f}, SD = {female.std():.3f}, n = {len(female)}")
        print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
        print(f"  Cohen's d = {cohen_d:.3f}")
        print(f"  Result: {'Significant difference' if p_val < 0.05 else 'No significant difference'} at α = 0.05")

# ==============================================================================
# 4. COMPARISON BY INSTITUTION TYPE
# ==============================================================================
print("\n" + "="*80)
print("4. COMPARISON BY INSTITUTION TYPE")
print("="*80)

print("\n4.1 Descriptive Statistics by Institution Type:")
inst_stats = df.groupby('Type of Institution:')[['Knowledge_Score', 'Practice_Score', 
                                                   'Attitude_Score', 'Overall_Cybersecurity_Score']].agg(['mean', 'std', 'count'])
print(inst_stats)

# T-test between University and Polytechnic
print("\n4.2 Independent T-Tests (University vs Polytechnic):")

university = df[df['Type of Institution:'] == 'University']
polytechnic = df[df['Type of Institution:'] == 'Polytechnic (Institute of Technology)/Tertiary']

for score, name in zip(scores_to_test, score_names):
    uni = university[score].dropna()
    poly = polytechnic[score].dropna()
    
    if len(uni) > 0 and len(poly) > 0:
        t_stat, p_val = stats.ttest_ind(uni, poly)
        cohen_d = (uni.mean() - poly.mean()) / np.sqrt(((len(uni)-1)*uni.std()**2 + (len(poly)-1)*poly.std()**2) / (len(uni)+len(poly)-2))
        
        print(f"\n{name} Score:")
        print(f"  University: Mean = {uni.mean():.3f}, SD = {uni.std():.3f}, n = {len(uni)}")
        print(f"  Polytechnic: Mean = {poly.mean():.3f}, SD = {poly.std():.3f}, n = {len(poly)}")
        print(f"  t-statistic = {t_stat:.3f}, p-value = {p_val:.4f}")
        print(f"  Cohen's d = {cohen_d:.3f}")
        print(f"  Result: {'Significant difference' if p_val < 0.05 else 'No significant difference'} at α = 0.05")

# ==============================================================================
# 5. COMPARISON BY PROGRAMME OF STUDY
# ==============================================================================
print("\n" + "="*80)
print("5. COMPARISON BY PROGRAMME OF STUDY")
print("="*80)

print("\n5.1 Descriptive Statistics by Programme:")
prog_stats = df.groupby('Programme of study:')[['Knowledge_Score', 'Practice_Score', 
                                                  'Attitude_Score', 'Overall_Cybersecurity_Score']].agg(['mean', 'std', 'count'])
print(prog_stats)

# ANOVA for programme types
print("\n5.2 One-Way ANOVA (Programme of Study):")

prog_categories = df['Programme of study:'].dropna().unique()

for score, name in zip(scores_to_test, score_names):
    groups = [df[df['Programme of study:'] == prog][score].dropna() for prog in prog_categories]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        
        print(f"\n{name} Score:")
        print(f"  F-statistic = {f_stat:.3f}, p-value = {p_val:.4f}")
        print(f"  Result: {'Significant difference' if p_val < 0.05 else 'No significant difference'} between programmes at α = 0.05")

# ==============================================================================
# 6. VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("6. GENERATING GROUP COMPARISON VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. Training Status - Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
training_means = df.groupby('Training_Binary')[['Knowledge_Score', 'Practice_Score', 'Attitude_Score']].mean()
x_pos = np.arange(len(training_means.index))
width = 0.25
ax1.bar(x_pos - width, training_means['Knowledge_Score'], width, label='Knowledge', color='steelblue', edgecolor='black')
ax1.bar(x_pos, training_means['Practice_Score'], width, label='Practice', color='coral', edgecolor='black')
ax1.bar(x_pos + width, training_means['Attitude_Score'], width, label='Attitude', color='lightgreen', edgecolor='black')
ax1.set_xlabel('Training Status (0=No, 1=Yes)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax1.set_title('Scores by Training Status', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(['No Training', 'Has Training'])
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Training Status - Box Plot
ax2 = fig.add_subplot(gs[0, 1])
training_data = [df[df['Training_Binary'] == 0]['Overall_Cybersecurity_Score'].dropna(),
                 df[df['Training_Binary'] == 1]['Overall_Cybersecurity_Score'].dropna()]
bp = ax2.boxplot(training_data, labels=['No Training', 'Has Training'], patch_artist=True,
                 showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], ['coral', 'lightgreen']):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)
ax2.set_ylabel('Overall Cybersecurity Score', fontsize=11, fontweight='bold')
ax2.set_title('Overall Score Distribution by Training', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Age Group - Overall Score
ax3 = fig.add_subplot(gs[0, 2])
age_means = df.groupby('Age group:')['Overall_Cybersecurity_Score'].mean().sort_values()
bars = ax3.barh(range(len(age_means)), age_means.values, color='skyblue', edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(age_means)))
ax3.set_yticklabels(age_means.index)
ax3.set_xlabel('Average Overall Score', fontsize=11, fontweight='bold')
ax3.set_title('Overall Score by Age Group', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, age_means.values)):
    ax3.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')

# 4. Gender - Bar Chart
ax4 = fig.add_subplot(gs[1, 0])
gender_means = df.groupby('Gender:')[['Knowledge_Score', 'Practice_Score', 'Attitude_Score']].mean()
gender_means.plot(kind='bar', ax=ax4, rot=45, color=['steelblue', 'coral', 'lightgreen'], 
                  edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Gender', fontsize=11, fontweight='bold')
ax4.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax4.set_title('Scores by Gender', fontsize=12, fontweight='bold')
ax4.legend(['Knowledge', 'Practice', 'Attitude'])
ax4.grid(True, alpha=0.3, axis='y')

# 5. Institution Type - Bar Chart
ax5 = fig.add_subplot(gs[1, 1])
inst_means = df.groupby('Type of Institution:')[['Knowledge_Score', 'Practice_Score', 'Attitude_Score']].mean()
inst_means.plot(kind='bar', ax=ax5, rot=45, color=['steelblue', 'coral', 'lightgreen'],
                edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Institution Type', fontsize=11, fontweight='bold')
ax5.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax5.set_title('Scores by Institution Type', fontsize=12, fontweight='bold')
ax5.legend(['Knowledge', 'Practice', 'Attitude'])
ax5.grid(True, alpha=0.3, axis='y')

# 6. Programme - Overall Score
ax6 = fig.add_subplot(gs[1, 2])
prog_means = df.groupby('Programme of study:')['Overall_Cybersecurity_Score'].mean().sort_values()
bars = ax6.barh(range(len(prog_means)), prog_means.values, color='gold', edgecolor='black', linewidth=1.5)
ax6.set_yticks(range(len(prog_means)))
ax6.set_yticklabels(prog_means.index)
ax6.set_xlabel('Average Overall Score', fontsize=11, fontweight='bold')
ax6.set_title('Overall Score by Programme', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, prog_means.values)):
    ax6.text(val, i, f' {val:.2f}', va='center', fontsize=10, fontweight='bold')

# 7. Training vs Non-Training - Violin Plot
ax7 = fig.add_subplot(gs[2, 0])
training_violin_data = pd.DataFrame({
    'Score': list(df[df['Training_Binary'] == 0]['Overall_Cybersecurity_Score'].dropna()) + 
             list(df[df['Training_Binary'] == 1]['Overall_Cybersecurity_Score'].dropna()),
    'Training': ['No Training']*len(df[df['Training_Binary'] == 0]['Overall_Cybersecurity_Score'].dropna()) +
                ['Has Training']*len(df[df['Training_Binary'] == 1]['Overall_Cybersecurity_Score'].dropna())
})
sns.violinplot(data=training_violin_data, x='Training', y='Score', ax=ax7, palette=['coral', 'lightgreen'])
ax7.set_ylabel('Overall Cybersecurity Score', fontsize=11, fontweight='bold')
ax7.set_xlabel('')
ax7.set_title('Score Distribution: Violin Plot', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Age Groups - Multiple Scores Comparison
ax8 = fig.add_subplot(gs[2, 1])
age_all_scores = df.groupby('Age group:')[['Knowledge_Score', 'Practice_Score', 'Attitude_Score']].mean()
x_pos = np.arange(len(age_all_scores.index))
width = 0.25
ax8.bar(x_pos - width, age_all_scores['Knowledge_Score'], width, label='Knowledge', 
        color='steelblue', edgecolor='black')
ax8.bar(x_pos, age_all_scores['Practice_Score'], width, label='Practice', 
        color='coral', edgecolor='black')
ax8.bar(x_pos + width, age_all_scores['Attitude_Score'], width, label='Attitude', 
        color='lightgreen', edgecolor='black')
ax8.set_xticks(x_pos)
ax8.set_xticklabels(age_all_scores.index, rotation=45, ha='right')
ax8.set_ylabel('Average Score', fontsize=11, fontweight='bold')
ax8.set_title('All Scores by Age Group', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 9. Effect Sizes - Training Impact
ax9 = fig.add_subplot(gs[2, 2])
effect_sizes = []
score_labels = []
for score, name in zip(scores_to_test, score_names):
    group1 = df[df['Training_Binary'] == 1][score].dropna()
    group2 = df[df['Training_Binary'] == 0][score].dropna()
    cohen_d = (group1.mean() - group2.mean()) / np.sqrt(((len(group1)-1)*group1.std()**2 + 
                                                          (len(group2)-1)*group2.std()**2) / 
                                                         (len(group1)+len(group2)-2))
    effect_sizes.append(cohen_d)
    score_labels.append(name)

colors_effect = ['lightgreen' if abs(d) >= 0.2 else 'coral' for d in effect_sizes]
bars = ax9.barh(score_labels, effect_sizes, color=colors_effect, edgecolor='black', linewidth=1.5)
ax9.axvline(x=0, color='black', linestyle='-', linewidth=2)
ax9.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Small effect')
ax9.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Medium effect')
ax9.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax9.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax9.set_xlabel("Cohen's d (Effect Size)", fontsize=11, fontweight='bold')
ax9.set_title('Training Effect Sizes', fontsize=12, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars, effect_sizes)):
    ax9.text(val, i, f' {val:.3f}', va='center', fontsize=10, fontweight='bold')

plt.savefig('group_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Group comparison visualizations saved as 'group_comparison.png'")

# ==============================================================================
# 7. SUMMARY OF KEY FINDINGS
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)

trained_overall = df[df['Training_Binary'] == 1]['Overall_Cybersecurity_Score'].mean()
not_trained_overall = df[df['Training_Binary'] == 0]['Overall_Cybersecurity_Score'].mean()
training_t, training_p = stats.ttest_ind(
    df[df['Training_Binary'] == 1]['Overall_Cybersecurity_Score'].dropna(),
    df[df['Training_Binary'] == 0]['Overall_Cybersecurity_Score'].dropna()
)

age_groups_list = [df[df['Age group:'] == age]['Overall_Cybersecurity_Score'].dropna() 
                   for age in df['Age group:'].unique() if pd.notna(age)]
age_f, age_p = stats.f_oneway(*age_groups_list)

male_scores = df[df['Gender:'] == 'Male']['Overall_Cybersecurity_Score'].dropna()
female_scores = df[df['Gender:'] == 'Female']['Overall_Cybersecurity_Score'].dropna()
if len(male_scores) > 0 and len(female_scores) > 0:
    gender_t, gender_p = stats.ttest_ind(male_scores, female_scores)
else:
    gender_p = 1.0

print(f"""
1. TRAINING STATUS:
   - Participants with training: {trained_overall:.3f}
   - Participants without training: {not_trained_overall:.3f}
   - Difference: {trained_overall - not_trained_overall:.3f}
   - Statistical significance: p = {training_p:.4f} ({'Significant' if training_p < 0.05 else 'Not significant'})

2. AGE GROUP:
   - ANOVA F-statistic: {age_f:.3f}
   - Statistical significance: p = {age_p:.4f} ({'Significant' if age_p < 0.05 else 'Not significant'})
   - Highest scoring age group: {df.groupby('Age group:')['Overall_Cybersecurity_Score'].mean().idxmax()}
   - Lowest scoring age group: {df.groupby('Age group:')['Overall_Cybersecurity_Score'].mean().idxmin()}

3. GENDER:
   - Male average: {male_scores.mean():.3f}
   - Female average: {female_scores.mean():.3f}
   - Statistical significance: p = {gender_p:.4f} ({'Significant' if gender_p < 0.05 else 'Not significant'})

4. INSTITUTION TYPE:
   - University: {df[df['Type of Institution:'] == 'University']['Overall_Cybersecurity_Score'].mean():.3f}
   - Polytechnic: {df[df['Type of Institution:'] == 'Polytechnic (Institute of Technology)/Tertiary']['Overall_Cybersecurity_Score'].mean():.3f}
""")

print("\n" + "="*80)
print("GROUP COMPARISON ANALYSIS COMPLETE")
print("="*80)