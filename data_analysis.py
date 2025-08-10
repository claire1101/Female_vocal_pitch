# Data analysis

# Load packages
import pandas as pd
import numpy as np
import unicodedata
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load dataset
df = pd.read_csv("/Path/to/female_candidates_with_optional_pitch.csv")

# Defining binary outcome variable: Elected (1) vs Not Elected (0)
df['elected_binary'] = df['Elected in GE24'].map({'Y': 1, 'N': 0})

# Quick descriptive analysis:

# Show missing pitch stats
print(df['mean_pitch_hz'].isna().sum(), "rows have missing pitch data")

# Compute stats only on available data
pitch_with_data = df[~df['mean_pitch_hz'].isna()]

# Separate groups
elected = pitch_with_data[pitch_with_data['Elected in GE24'] == 'Y']['mean_pitch_hz']
not_elected = pitch_with_data[pitch_with_data['Elected in GE24'] == 'N']['mean_pitch_hz']

# T-test ignoring NaNs
t_stat, p_val = ttest_ind(elected, not_elected, nan_policy='omit', equal_var=False)
print(f"T-test p-value: {p_val:.4f}")

# Histograms by election outcome
plt.figure(figsize=(8, 4))
pitch_with_data['mean_pitch_hz'].hist(by=pitch_with_data['Elected in GE24'].map({'Y':1,'N':0}), bins=15)
plt.show()

# Getting summary statistics
print("Elected pitch stats:\n", elected.describe())
print("Not elected pitch stats:\n", not_elected.describe())

# Making a boxplot
sns.boxplot(x='Elected in GE24', y='mean_pitch_hz', data=pitch_with_data)
plt.show()

# Creating a binary variable for incumbency
# 1 = incumbent, 0 = not
df['incumbent_binary'] = df['Incumbency'].map({'Y': 1, 'N': 0})

# Group political parties by ideology
party_ideology_map = {
    # Center
    "Fianna Fáil": "Center",
    "Fine Gael": "Center",
    "Aontú": "Center",

    # Left
    "Sinn Féin": "Left",
    "Labour": "Left",
    "Green Party": "Left",
    "Social Democrats": "Left",
    "Solidarity PBP": "Left",
    "People Before Profit": "Left",
    "Socialist Party": "Left",

    # Right
    "Irish Freedom Party": "Right",
    "National Party": "Right",
    "The Irish People": "Right",
    "Catholic Democrats": "Right",

    # Other / Independents
    "Independent": "Independent",
    "Independent Ireland": "Independent",
    "The Irish People Movement": "Independent",
    "Party for Animal Welfare": "Independent",
    "Liberty Republic": "Independent",
    "Irish Democratic Party": "Independent",
    "Rabharta": "Independent",
    "Éirígí": "Independent",
    "Workers Party": "Independent",
    "Independent Left": "Independent",
    "United Left": "Independent",
    "Direct Democracy Ireland": "Independent",
    "An Rabharta Glas – Green Left": "Independent",
}
df["ideology"] = df["Party Full"].map(party_ideology_map).fillna("Independent")

# Check if any parties were not mapped
unmapped = df[df['ideology'].isnull()]['Party Full'].unique()
if len(unmapped) > 0:
    print("Unmapped parties:", unmapped)

# Create dummies for ideology without touching original data
ideology_dummies = pd.get_dummies(df['ideology'], prefix='ideology', drop_first=True)

# Prepare regression dataframe
predictors = pd.concat([df[['mean_pitch_hz', 'incumbent_binary']], ideology_dummies], axis=1)

# Drop only rows where pitch is missing
regression_data = predictors.dropna(subset=['mean_pitch_hz']).copy()

# Add outcome variable and interaction term
regression_data['elected_binary'] = df.loc[regression_data.index, 'elected_binary']
regression_data['pitch_incumbent_interaction'] = regression_data['mean_pitch_hz'] * regression_data['incumbent_binary']

# Add constant
X = sm.add_constant(regression_data.drop(columns=['elected_binary']))
y = regression_data['elected_binary']

# Debugging: Check for non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("Non-numeric columns found in X:", list(non_numeric_cols))
else:
    print("All predictor columns are numeric.")

# Identify and convert ideology columns to integers
ideology_cols = [col for col in X.columns if col.startswith('ideology_')]
X[ideology_cols] = X[ideology_cols].astype(int)

# Make sure the rest of X is numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Double-check all columns are numeric
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print("Still non-numeric columns in X:", list(non_numeric_cols))
else:
    print("All columns in X are now numeric.")

# Ensure y is numeric too
y = pd.to_numeric(y, errors='coerce')

# Drop any remaining NaNs from X or y
valid_idx = X.dropna().index.intersection(y.dropna().index)
X = X.loc[valid_idx]
y = y.loc[valid_idx]

# Drop ideology_Right from the predictors to avoid perfect multicollinearity 
X = X.drop(columns=['ideology_Right'])

# Fit logistic regression model with robust standard errors
model = sm.Logit(y, X).fit(cov_type='HC3')
print(model.summary())

# Exponentiate coefficients to get odds ratios
odds_ratios = np.exp(model.params)
print("Odds Ratios:\n", odds_ratios)

# Predicted probability plot

# Create a grid of pitch values
pitch_range = np.linspace(df['mean_pitch_hz'].min(), df['mean_pitch_hz'].max(), 100)

# Predict probabilities for incumbents and non-incumbents
def make_prediction(incumbent):
    temp_df = pd.DataFrame({
        'const': 1,
        'mean_pitch_hz': pitch_range,
        'incumbent_binary': incumbent,
        'ideology_Independent': 0,  # set baseline ideology
        'ideology_Left': 0,
        'pitch_incumbent_interaction': pitch_range * incumbent
    })
    return model.predict(temp_df)

prob_incumbent = make_prediction(1)
prob_non_incumbent = make_prediction(0)

plt.figure(figsize=(10, 6))
plt.plot(pitch_range, prob_incumbent, label='Incumbent', color='blue')
plt.plot(pitch_range, prob_non_incumbent, label='Non-incumbent', color='orange')
plt.xlabel('Mean Pitch (Hz)')
plt.ylabel('Predicted Probability of Election')
plt.title('Effect of Vocal Pitch on Election Probability')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Robustness checks

# Variance inflation factor
# X should not include the dependent variable
X_vif = X.copy()
X_vif = X_vif.drop(columns=['const'])  # remove constant for VIF

# Create a DataFrame of VIFs
vif_df = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})

print(vif_df)

# Compare w/ a reduced model
# Define reduced predictor matrix: only pitch and incumbency (no ideology, no interaction)
X_reduced = regression_data[['mean_pitch_hz', 'incumbent_binary']]

# Add constant
X_reduced = sm.add_constant(X_reduced)

# Ensure all columns are numeric
X_reduced = X_reduced.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaNs that may have crept in
X_reduced = X_reduced.dropna()
y_reduced = regression_data.loc[X_reduced.index, 'elected_binary']

# Fit reduced model
reduced_model = sm.Logit(y_reduced, X_reduced).fit(cov_type='HC3')
print(reduced_model.summary())

# Sensitivity to exclusion of ideology variables
# Pitch and incumbency only
X_pitch_only = regression_data[['mean_pitch_hz', 'incumbent_binary', 'pitch_incumbent_interaction']]
X_pitch_only = sm.add_constant(X_pitch_only)

pitch_only_model = sm.Logit(y, X_pitch_only).fit(cov_type='HC3')
print(pitch_only_model.summary())

# Getting descriptive stats for key variables

# Descriptive stats for mean pitch
pitch_desc = df['mean_pitch_hz'].describe()
print("\n Mean Pitch (Hz):")
print(f"  Mean: {pitch_desc['mean']:.2f}")
print(f"  Min: {pitch_desc['min']:.2f}")
print(f"  Max: {pitch_desc['max']:.2f}")
print(f"  SD:  {pitch_desc['std']:.2f}")

# Proportion of incumbents
incumbent_rate = df['incumbent_binary'].mean()
print("\n Incumbency:")
print(f"  Proportion of incumbents: {incumbent_rate:.2%}")

# Proportion elected
elected_rate = df['elected_binary'].mean()
print("\n Election Outcome:")
print(f"  Proportion elected: {elected_rate:.2%}")

# Ideology distribution
print("\n Ideological Distribution:")
print(df['ideology'].value_counts())

# More descriptive statistics

# Basic stats overall
overall_stats = df['mean_pitch_hz'].agg(['mean', 'std', 'min', 'max'])

# Grouped descriptive stats
grouped_stats = df.groupby('incumbent_binary')['mean_pitch_hz'].describe()
elected_stats = df.groupby('elected_binary')['mean_pitch_hz'].describe()
ideology_stats = df.groupby('ideology')['mean_pitch_hz'].describe()

# Display stats
print("Overall Pitch Stats:\n", overall_stats)
print("\nBy Incumbency:\n", grouped_stats)
print("\nBy Election Outcome:\n", elected_stats)
print("\nBy Ideology:\n", ideology_stats)

# Vocal pitch range per group
range_by_incumbency = df.groupby('incumbent_binary')['mean_pitch_hz'].agg(lambda x: x.max() - x.min())
range_by_elected = df.groupby('elected_binary')['mean_pitch_hz'].agg(lambda x: x.max() - x.min())
range_by_ideology = df.groupby('ideology')['mean_pitch_hz'].agg(lambda x: x.max() - x.min())

print("\nPitch Range by Incumbency:\n", range_by_incumbency)
print("\nPitch Range by Election Outcome:\n", range_by_elected)
print("\nPitch Range by Ideology:\n", range_by_ideology)

# Boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(x='incumbent_binary', y='mean_pitch_hz', data=df)
plt.title('Vocal Pitch by Incumbency')
plt.xlabel('Incumbent (0 = No, 1 = Yes)')
plt.ylabel('Mean Pitch (Hz)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='elected_binary', y='mean_pitch_hz', data=df)
plt.title('Vocal Pitch by Election Outcome')
plt.xlabel('Elected (0 = No, 1 = Yes)')
plt.ylabel('Mean Pitch (Hz)')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='ideology', y='mean_pitch_hz', data=df)
plt.title('Vocal Pitch by Political Ideology')
plt.xlabel('Ideology')
plt.ylabel('Mean Pitch (Hz)')
plt.xticks(rotation=45)
plt.show()

# Histogram with KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['mean_pitch_hz'], kde=True, bins=20)
plt.title('Distribution of Mean Vocal Pitch (Hz)')
plt.xlabel('Mean Pitch (Hz)')
plt.ylabel('Count')
plt.show()

# Running an additional, bivariate model

# Drop rows with missing or infinite values in the relevant columns
df_clean = df[['mean_pitch_hz', 'elected_binary']].replace([np.inf, -np.inf], np.nan).dropna()

# Separate variables
X = df_clean[['mean_pitch_hz']]
y = df_clean['elected_binary']

# Add constant for intercept
X = sm.add_constant(X)

# Logistic regression
model = sm.Logit(y, X).fit()

# Summary of results
print(model.summary())

# Odds ratios
odds_ratios = pd.DataFrame({
    "Variable": X.columns,
    "Odds Ratio": np.exp(model.params).round(3)
})
print("\nOdds Ratios:")
print(odds_ratios)