import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load the data
df = pd.read_csv('cybersecurity_survey.csv')

print("="*80)
print("DESCRIPTIVE STATISTICS - CYBERSECURITY SURVEY")
print("="*80)

# Clean column names
df.columns = df.columns.str.strip()

# ==============================================================================
# 1. DATASET OVERVIEW
# ==============================================================================
print("\n1. DATASET OVERVIEW")
print("-"*80)
print(f"Total Responses: {len(df)}")
print(f"Total Variables: {len(df.columns)}")
print(f"Missing Values: {df.isnull().sum().sum()}")

print("\nColumn Names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

# ==============================================================================
# 2. DEMOGRAPHIC ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("2. DEMOGRAPHIC DISTRIBUTION")
print("="*80)

print("\n2.1 Age Group Distribution")
age_dist = df['Age group:'].value_counts().sort_index()
print(age_dist)
print(f"\nPercentages:")
print((age_dist / len(df) * 100).round(2))

print("\n2.2 Gender Distribution")
gender_dist = df['Gender:'].value_counts()
print(gender_dist)
print(f"\nPercentages:")
print((gender_dist / len(df) * 100).round(2))

print("\n2.3 Institution Type")
inst_dist = df['Type of Institution:'].value_counts()
print(inst_dist)
print(f"\nPercentages:")
print((inst_dist / len(df) * 100).round(2))

print("\n2.4 Programme of Study")
prog_dist = df['Programme of study:'].value_counts()
print(prog_dist)
print(f"\nPercentages:")
print((prog_dist / len(df) * 100).round(2))

print("\n2.5 Year of Study")
year_dist = df['Year of study:'].value_counts()
print(year_dist)

# ==============================================================================
# 3. KNOWLEDGE SCORES ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("3. KNOWLEDGE SCORES (1-5 Likert Scale)")
print("="*80)

knowledge_cols = [
    'I understand what phishing is.',
    'I can recognise a phishing email or message when I see one.',
    'I know what two-factor authentication (2FA) is and why it is used.',
    'I understand what malware is and how it functions.',
    'I am familiar with what a Virtual Private Network (VPN) does.',
    'I know best practices for creating strong and secure passwords.',
    'I understand the risks of using public Wi-Fi without additional protections.'
]

knowledge_short = [
    'Phishing Understanding',
    'Phishing Recognition',
    '2FA Knowledge',
    'Malware Understanding',
    'VPN Familiarity',
    'Password Best Practices',
    'Public WiFi Risks'
]

print("\nKnowledge Items Statistics:")
knowledge_stats = df[knowledge_cols].describe()
print(knowledge_stats)

print("\nIndividual Knowledge Item Means:")
for col, short in zip(knowledge_cols, knowledge_short):
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"{short}: {mean_val:.2f} ± {std_val:.2f}")

# ==============================================================================
# 4. PRACTICE SCORES ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("4. PRACTICE SCORES (1-5 Likert Scale)")
print("="*80)

practice_cols = [
    'I use a unique password for each of my important online accounts.',
    'I regularly update the software and applications on my devices.',
    'I enable two-factor authentication (2FA) on accounts whenever possible.',
    'I avoid clicking links or opening attachments from unknown or suspicious emails.',
    'I review app permissions and only allow access to what is necessary.',
    'I back up important files regularly to a secure location.',
    'I use a VPN or other protections when accessing public Wi-Fi for sensitive tasks.'
]

practice_short = [
    'Unique Passwords',
    'Software Updates',
    '2FA Enabled',
    'Avoid Suspicious Links',
    'App Permissions Review',
    'Regular Backups',
    'VPN Usage'
]

print("\nPractice Items Statistics:")
practice_stats = df[practice_cols].describe()
print(practice_stats)

print("\nIndividual Practice Item Means:")
for col, short in zip(practice_cols, practice_short):
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"{short}: {mean_val:.2f} ± {std_val:.2f}")

# ==============================================================================
# 5. ATTITUDE SCORES ANALYSIS
# ==============================================================================
print("\n" + "="*80)
print("5. ATTITUDE SCORES (1-5 Likert Scale)")
print("="*80)

attitude_cols = [
    'I feel confident in my ability to detect online scams and phishing attempts.',
    'I am concerned about the privacy and security of my personal information online.',
    'I believe cybersecurity knowledge is important for my academic and professional life.',
    'I believe my current online practices are sufficient to protect me from most cyber threats.',
    'I would attend a workshop on practical cybersecurity skills if my institution offered one.'
]

attitude_short = [
    'Confidence in Detection',
    'Privacy Concern',
    'Importance Belief',
    'Sufficiency Belief',
    'Workshop Willingness'
]

print("\nAttitude Items Statistics:")
attitude_stats = df[attitude_cols].describe()
print(attitude_stats)

print("\nIndividual Attitude Item Means:")
for col, short in zip(attitude_cols, attitude_short):
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"{short}: {mean_val:.2f} ± {std_val:.2f}")

# ==============================================================================
# 6. COMPOSITE SCORES
# ==============================================================================
print("\n" + "="*80)
print("6. COMPOSITE SCORES")
print("="*80)

df['Knowledge_Score'] = df[knowledge_cols].mean(axis=1)
df['Practice_Score'] = df[practice_cols].mean(axis=1)
df['Attitude_Score'] = df[attitude_cols].mean(axis=1)
df['Overall_Cybersecurity_Score'] = (df['Knowledge_Score'] + df['Practice_Score'] + df['Attitude_Score']) / 3

composite_summary = df[['Knowledge_Score', 'Practice_Score', 'Attitude_Score', 'Overall_Cybersecurity_Score']].describe()
print(composite_summary)

print("\nComposite Score Means:")
print(f"Knowledge Score: {df['Knowledge_Score'].mean():.3f} ± {df['Knowledge_Score'].std():.3f}")
print(f"Practice Score: {df['Practice_Score'].mean():.3f} ± {df['Practice_Score'].std():.3f}")
print(f"Attitude Score: {df['Attitude_Score'].mean():.3f} ± {df['Attitude_Score'].std():.3f}")
print(f"Overall Cybersecurity Score: {df['Overall_Cybersecurity_Score'].mean():.3f} ± {df['Overall_Cybersecurity_Score'].std():.3f}")

# ==============================================================================
# 7. TRAINING & INSTITUTIONAL SUPPORT
# ==============================================================================
print("\n" + "="*80)
print("7. TRAINING & INSTITUTIONAL SUPPORT")
print("="*80)

print("\n7.1 Formal Cybersecurity Training Received:")
training_dist = df['I have received formal cybersecurity training (e.g., course, workshop, or module) during my studies.'].value_counts()
print(training_dist)
print(f"\nPercentages:")
print((training_dist / len(df) * 100).round(2))

print("\n7.2 Institution Provides Adequate Information (1-5 scale):")
inst_info = df['My institution provides adequate information on how to stay safe online.']
print(inst_info.value_counts().sort_index())
print(f"Mean: {inst_info.mean():.2f} ± {inst_info.std():.2f}")

print("\n7.3 Preferred Learning Methods:")
learning_pref = df['I prefer to learn about cybersecurity through:'].value_counts()
print(learning_pref.head(10))

# ==============================================================================
# 8. VISUALIZATIONS
# ==============================================================================
print("\n" + "="*80)
print("8. GENERATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Demographics - Age
ax1 = fig.add_subplot(gs[0, 0])
age_counts = df['Age group:'].value_counts()
ax1.bar(range(len(age_counts)), age_counts.values, color='steelblue')
ax1.set_xticks(range(len(age_counts)))
ax1.set_xticklabels(age_counts.index, rotation=45, ha='right')
ax1.set_ylabel('Count')
ax1.set_title('Age Group Distribution', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Demographics - Gender
ax2 = fig.add_subplot(gs[0, 1])
gender_counts = df['Gender:'].value_counts()
ax2.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Gender Distribution', fontweight='bold')

# 3. Programme Distribution
ax3 = fig.add_subplot(gs[0, 2])
prog_counts = df['Programme of study:'].value_counts()
ax3.barh(range(len(prog_counts)), prog_counts.values, color='coral')
ax3.set_yticks(range(len(prog_counts)))
ax3.set_yticklabels(prog_counts.index)
ax3.set_xlabel('Count')
ax3.set_title('Programme of Study', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Knowledge Items
ax4 = fig.add_subplot(gs[1, 0])
knowledge_means = [df[col].mean() for col in knowledge_cols]
ax4.barh(range(len(knowledge_short)), knowledge_means, color='lightgreen')
ax4.set_yticks(range(len(knowledge_short)))
ax4.set_yticklabels(knowledge_short, fontsize=9)
ax4.set_xlabel('Mean Score (1-5)')
ax4.set_xlim(0, 5)
ax4.set_title('Knowledge Items - Mean Scores', fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Practice Items
ax5 = fig.add_subplot(gs[1, 1])
practice_means = [df[col].mean() for col in practice_cols]
ax5.barh(range(len(practice_short)), practice_means, color='lightcoral')
ax5.set_yticks(range(len(practice_short)))
ax5.set_yticklabels(practice_short, fontsize=9)
ax5.set_xlabel('Mean Score (1-5)')
ax5.set_xlim(0, 5)
ax5.set_title('Practice Items - Mean Scores', fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# 6. Attitude Items
ax6 = fig.add_subplot(gs[1, 2])
attitude_means = [df[col].mean() for col in attitude_cols]
ax6.barh(range(len(attitude_short)), attitude_means, color='lightskyblue')
ax6.set_yticks(range(len(attitude_short)))
ax6.set_yticklabels(attitude_short, fontsize=9)
ax6.set_xlabel('Mean Score (1-5)')
ax6.set_xlim(0, 5)
ax6.set_title('Attitude Items - Mean Scores', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='x')

# 7. Composite Scores Distribution
ax7 = fig.add_subplot(gs[2, 0])
composite_data = [df['Knowledge_Score'], df['Practice_Score'], df['Attitude_Score'], df['Overall_Cybersecurity_Score']]
bp = ax7.boxplot(composite_data, labels=['Knowledge', 'Practice', 'Attitude', 'Overall'], patch_artist=True)
colors = ['lightgreen', 'lightcoral', 'lightskyblue', 'gold']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax7.set_ylabel('Score')
ax7.set_title('Composite Scores Distribution', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Training Status
ax8 = fig.add_subplot(gs[2, 1])
training_counts = df['I have received formal cybersecurity training (e.g., course, workshop, or module) during my studies.'].value_counts()
ax8.pie(training_counts.values, labels=training_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'], startangle=90)
ax8.set_title('Formal Training Received', fontweight='bold')

# 9. Composite Scores Comparison
ax9 = fig.add_subplot(gs[2, 2])
composite_means = [df['Knowledge_Score'].mean(), df['Practice_Score'].mean(), df['Attitude_Score'].mean()]
composite_labels = ['Knowledge', 'Practice', 'Attitude']
bars = ax9.bar(composite_labels, composite_means, color=['lightgreen', 'lightcoral', 'lightskyblue'])
ax9.set_ylabel('Mean Score')
ax9.set_ylim(0, 5)
ax9.set_title('Average Composite Scores', fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

plt.savefig('descriptive_statistics.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'descriptive_statistics.png'")

# Save composite scores for other analyses
df.to_csv('survey_data_with_scores.csv', index=False)
print("✓ Data with composite scores saved as 'survey_data_with_scores.csv'")

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS ANALYSIS COMPLETE")
print("="*80)