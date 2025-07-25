
# The IPL Data Analysis Case Study 
- involves two datasets: `matches_data.csv` and `deliveries_data.csv`, which provide comprehensive information about Indian Premier League (IPL) matches and ball-by-ball details, respectively. 
- Below, I will explain the datasets, provide basic and advanced questions for analysis, and include a sample artifact for one of the questions.

### Explanation of the Datasets

#### 1. `matches_data.csv`
This dataset contains match-level information for IPL matches. Each row represents a single match, with details about the teams, venue, date, and match outcome. Based on typical IPL match datasets, the likely columns include:
- **id**: Unique identifier for the match.
- **season**: The year or season of the IPL (e.g., 2008, 2009).
- **city**: City where the match was played.
- **date**: Date of the match.
- **team1**, **team2**: The two teams playing the match.
- **toss_winner**: Team that won the toss.
- **toss_decision**: Decision to bat or field after winning the toss.
- **result**: Outcome of the match (normal, tie, no result).
- **winner**: Winning team.
- **player_of_match**: Player awarded as the match's best performer.
- **venue**: Stadium or ground where the match was played.
- **umpire1**, **umpire2**: Names of the umpires.
- **win_by_runs**, **win_by_wickets**: Margin of victory (runs or wickets).

#### 2. `deliveries_data.csv`
This dataset contains ball-by-ball details for each match, providing granular data about each delivery bowled. Likely columns include:
- **match_id**: Identifier linking to the match in `matches_data.csv`.
- **inning**: Inning number (1 or 2).
- **batting_team**: Team batting during the delivery.
- **bowling_team**: Team bowling during the delivery.
- **over**: Over number (1 to 20 in T20).
- **ball**: Ball number within the over (1 to 6, excluding extras).
- **batsman**: Player facing the delivery.
- **non_striker**: Batsman at the non-striker's end.
- **bowler**: Player bowling the delivery.
- **is_super_over**: Indicator if the delivery was in a super over (0 or 1).
- **wide_runs**, **bye_runs**, **legbye_runs**, **noball_runs**, **penalty_runs**: Runs scored via extras.
- **batsman_runs**: Runs scored by the batsman off the delivery.
- **total_runs**: Total runs scored on the delivery (batsman runs + extras).
- **player_dismissed**: Player dismissed on the delivery (if any).
- **dismissal_kind**: Type of dismissal (e.g., caught, bowled, run out).
- **fielder**: Fielder involved in the dismissal (if applicable).

### Basic Questions for Analysis
These questions are suitable for beginners and focus on simple aggregations and insights:
1. **Which team has won the most matches in the IPL?**
   - Analyze the `winner` column in `matches_data.csv` to count wins per team.
2. **What is the average win margin (by runs and wickets) for matches?**
   - Use `win_by_runs` and `win_by_wickets` from `matches_data.csv` to compute averages.
3. **Which player has won the most "Player of the Match" awards?**
   - Count occurrences in the `player_of_match` column in `matches_data.csv`.
4. **How many matches were played in each season?**
   - Group by `season` in `matches_data.csv` to count matches per season.
5. **Which venue hosted the most IPL matches?**
   - Analyze the `venue` column in `matches_data.csv` to count matches per venue.
6. **What is the total number of runs scored by each team in the IPL?**
   - Use `batting_team` and `total_runs` from `deliveries_data.csv` to sum runs per team.
7. **Which bowler has taken the most wickets in the IPL?**
   - Count dismissals in `player_dismissed` where the `bowler` is credited in `deliveries_data.csv`.
8. **What is the distribution of toss decisions (bat/field) across seasons?**
   - Group by `season` and `toss_decision` in `matches_data.csv`.

### Advanced Questions for Analysis
These questions require more complex analysis, such as joining datasets, calculating metrics, or performing time-series analysis:
1. **What is the batting strike rate of each batsman in the IPL, considering only players with at least 500 runs?**
   - Calculate strike rate as (total `batsman_runs` / balls faced) * 100, filtering for players with significant runs.
2. **Which team has the highest run rate in the powerplay overs (1–6) across all seasons?**
   - Filter `deliveries_data.csv` for overs 1–6, sum `total_runs`, and divide by overs faced per team.
3. **How does winning the toss impact match outcomes for each team?**
   - Join `matches_data.csv` (`toss_winner`, `winner`) and analyze win percentages when teams win/lose the toss.
4. **Which bowler has the best economy rate in death overs (16–20) with at least 50 overs bowled?**
   - Filter `deliveries_data.csv` for overs 16–20, calculate economy rate (runs conceded per over), and filter by overs bowled.
5. **What is the impact of batting first vs. second on match outcomes across seasons?**
   - Analyze `matches_data.csv` for `toss_decision` and `winner` to compare win rates for batting first vs. chasing.
6. **Which batsman has the highest average against a specific bowler (minimum 30 balls faced)?**
   - Join `batsman` and `bowler` in `deliveries_data.csv`, calculate average runs per dismissal.
7. **How do dismissal types (caught, bowled, run out, etc.) vary by batting team and season?**
   - Group `deliveries_data.csv` by `batting_team`, `dismissal_kind`, and `season` from `matches_data.csv`.
8. **Can we predict the likelihood of a team winning based on venue and toss decision using logistic regression?**
   - Build a model using `matches_data.csv` with features like `venue`, `toss_decision`, and target as `winner`.

### Sample Artifact: Python Code for Basic Question 1
Below is a Python script to answer the question: "Which team has won the most matches in the IPL?"

```python
import pandas as pd

# Load the matches dataset
matches_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/matches_data.csv')

# Count the number of wins per team
team_wins = matches_df['winner'].value_counts()

# Get the team with the most wins
most_wins_team = team_wins.idxmax()
most_wins_count = team_wins.max()

# Print the result
print(f"The team with the most IPL match wins is {most_wins_team} with {most_wins_count} wins.")

# Save the result to a CSV file
team_wins.to_csv('team_wins.csv')
print("Team wins data saved to 'team_wins.csv'.")
```



#### Requirements
-- To run this script, install the required libraries:

```python
pip install pandas numpy scikit-learn matplotlib
```

```python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the datasets
matches_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/matches_data.csv')
deliveries_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/deliveries_data.csv')

# Set up matplotlib style (use 'seaborn-v0_8' if available, else 'ggplot' or 'default')
available_styles = plt.style.available
style_to_use = 'seaborn-v0_8' if 'seaborn-v0_8' in available_styles else 'ggplot' if 'ggplot' in available_styles else 'default'
plt.style.use(style_to_use)
print(f"Using Matplotlib style: {style_to_use}")

# Question 1: Batting strike rate of each batsman (minimum 500 runs)
batsman_stats = deliveries_df.groupby('batsman').agg({
    'batsman_runs': 'sum',
    'ball': 'count'
})
batsman_stats['strike_rate'] = (batsman_stats['batsman_runs'] / batsman_stats['ball'] * 100).round(2)
batsman_stats = batsman_stats[batsman_stats['batsman_runs'] >= 500][['strike_rate']].sort_values('strike_rate', ascending=False)
print("Q1: Batting strike rates (min 500 runs):\n", batsman_stats.head())
batsman_stats.to_csv('batsman_strike_rates.csv')
print("Q1: Batting strike rates saved to 'batsman_strike_rates.csv'.")
# Plot top 10 batsmen by strike rate
plt.figure(figsize=(10, 6))
batsman_stats.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Batsmen by Strike Rate (Min 500 Runs)')
plt.xlabel('Batsman')
plt.ylabel('Strike Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('batsman_strike_rates.png')
plt.close()
print("Q1: Plot saved to 'batsman_strike_rates.png'.")

# Question 2: Team with highest run rate in powerplay overs (1–6)
powerplay_df = deliveries_df[deliveries_df['over'].between(1, 6)]
team_powerplay = powerplay_df.groupby('batting_team').agg({
    'total_runs': 'sum',
    'over': 'count'
})
team_powerplay['overs'] = team_powerplay['over'] / 6
team_powerplay['run_rate'] = (team_powerplay['total_runs'] / team_powerplay['overs']).round(2)
top_team = team_powerplay['run_rate'].idxmax()
top_run_rate = team_powerplay['run_rate'].max()
print(f"Q2: Team with highest powerplay run rate is {top_team} with {top_run_rate:.2f} runs per over.")
team_powerplay[['run_rate']].to_csv('powerplay_run_rates.csv')
print("Q2: Powerplay run rates saved to 'powerplay_run_rates.csv'.")
# Plot run rate by team
plt.figure(figsize=(10, 6))
team_powerplay['run_rate'].sort_values(ascending=False).plot(kind='bar', color='lightgreen')
plt.title('Powerplay Run Rate by Team')
plt.xlabel('Team')
plt.ylabel('Run Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('powerplay_run_rates.png')
plt.close()
print("Q2: Plot saved to 'powerplay_run_rates.png'.")

# Question 3: Impact of toss win on match outcomes
matches_df['toss_win_match_win'] = matches_df['toss_winner'] == matches_df['winner']
toss_impact = matches_df.groupby('toss_winner')['toss_win_match_win'].mean().round(2) * 100
print("Q3: Win percentage when winning toss:\n", toss_impact)
toss_impact.to_csv('toss_impact.csv')
print("Q3: Toss impact data saved to 'toss_impact.csv'.")
# Plot toss impact
plt.figure(figsize=(10, 6))
toss_impact.sort_values(ascending=False).plot(kind='bar', color='coral')
plt.title('Win Percentage When Winning Toss by Team')
plt.xlabel('Team')
plt.ylabel('Win Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('toss_impact.png')
plt.close()
print("Q3: Plot saved to 'toss_impact.png'.")

# Question 4: Bowler with best economy rate in death overs (16–20, min 50 overs)
death_overs_df = deliveries_df[deliveries_df['over'].between(16, 20)]
bowler_stats = death_overs_df.groupby('bowler').agg({
    'total_runs': 'sum',
    'over': 'count'
})
bowler_stats['overs'] = bowler_stats['over'] / 6
bowler_stats['economy_rate'] = (bowler_stats['total_runs'] / bowler_stats['overs']).round(2)
bowler_stats = bowler_stats[bowler_stats['overs'] >= 50][['economy_rate']].sort_values('economy_rate')
top_bowler = bowler_stats['economy_rate'].idxmin()
top_economy = bowler_stats['economy_rate'].min()
print(f"Q4: Bowler with best death overs economy is {top_bowler} with {top_economy:.2f} runs per over.")
bowler_stats.to_csv('death_overs_economy.csv')
print("Q4: Death overs economy rates saved to 'death_overs_economy.csv'.")
# Plot top 10 bowlers by economy rate
plt.figure(figsize=(10, 6))
bowler_stats.head(10).plot(kind='bar', color='lightblue')
plt.title('Top 10 Bowlers by Death Overs Economy Rate (Min 50 Overs)')
plt.xlabel('Bowler')
plt.ylabel('Economy Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('death_overs_economy.png')
plt.close()
print("Q4: Plot saved to 'death_overs_economy.png'.")

# Question 5: Impact of batting first vs. second on match outcomes
matches_df['batting_first'] = matches_df['toss_decision'].map({'bat': True, 'field': False})
matches_df['batting_first_won'] = matches_df.apply(
    lambda x: x['winner'] == x['team1'] if x['batting_first'] else x['winner'] == x['team2'], axis=1
)
batting_impact = matches_df.groupby('season')['batting_first_won'].mean().round(2) * 100
print("Q5: Win percentage when batting first:\n", batting_impact)
batting_impact.to_csv('batting_first_impact.csv')
print("Q5: Batting first impact saved to 'batting_first_impact.csv'.")
# Plot batting first win percentage
plt.figure(figsize=(10, 6))
batting_impact.plot(kind='line', marker='o', color='purple')
plt.title('Win Percentage When Batting First by Season')
plt.xlabel('Season')
plt.ylabel('Win Percentage (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('batting_first_impact.png')
plt.close()
print("Q5: Plot saved to 'batting_first_impact.png'.")

# Question 6: Batsman with highest average against a specific bowler (min 30 balls)
batsman_bowler = deliveries_df.groupby(['batsman', 'bowler']).agg({
    'batsman_runs': 'sum',
    'ball': 'count',
    'player_dismissed': lambda x: x.notna().sum()
})
batsman_bowler = batsman_bowler[batsman_bowler['ball'] >= 30]
batsman_bowler['batting_avg'] = (batsman_bowler['batsman_runs'] / batsman_bowler['player_dismissed'].replace(0, 1)).round(2)
top_batsman_bowler = batsman_bowler['batting_avg'].idxmax()
top_avg = batsman_bowler['batting_avg'].max()
print(f"Q6: Batsman with highest average against a bowler is {top_batsman_bowler[0]} against {top_batsman_bowler[1]} with {top_avg:.2f}.")
batsman_bowler[['batting_avg']].to_csv('batsman_bowler_avg.csv')
print("Q6: Batsman-bowler averages saved to 'batsman_bowler_avg.csv'.")
# Plot top 10 batsman-bowler pairs by average
plt.figure(figsize=(10, 6))
batsman_bowler['batting_avg'].head(10).plot(kind='bar', color='gold')
plt.title('Top 10 Batsman-Bowler Pairs by Batting Average (Min 30 Balls)')
plt.xlabel('Batsman-Bowler Pair')
plt.ylabel('Batting Average')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('batsman_bowler_avg.png')
plt.close()
print("Q6: Plot saved to 'batsman_bowler_avg.png'.")

# Question 7: Dismissal types by batting team and season
deliveries_with_season = deliveries_df.merge(matches_df[['id', 'season']], left_on='match_id', right_on='id')
dismissal_types = deliveries_with_season.groupby(['season', 'batting_team', 'dismissal_kind']).size().unstack(fill_value=0)
print("Q7: Dismissal types by team and season:\n", dismissal_types.head())
dismissal_types.to_csv('dismissal_types.csv')
print("Q7: Dismissal types saved to 'dismissal_types.csv'.")
# Plot dismissal types for a specific season (e.g., latest season)
latest_season = dismissal_types.index.get_level_values('season').max()
dismissal_types.xs(latest_season, level='season').sum().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ffb3e6'])
plt.title(f'Dismissal Types Distribution in Season {latest_season}')
plt.ylabel('')
plt.tight_layout()
plt.savefig('dismissal_types.png')
plt.close()
print("Q7: Plot saved to 'dismissal_types.png'.")

# Question 8: Logistic regression to predict team winning based on venue and toss decision
matches_df['team1_won'] = (matches_df['winner'] == matches_df['team1']).astype(int)
features = matches_df[['venue', 'toss_decision', 'team1', 'team2']]
le_venue = LabelEncoder()
le_toss = LabelEncoder()
le_team1 = LabelEncoder()
le_team2 = LabelEncoder()
features['venue'] = le_venue.fit_transform(features['venue'])
features['toss_decision'] = le_toss.fit_transform(features['toss_decision'])
features['team1'] = le_team1.fit_transform(features['team1'])
features['team2'] = le_team2.fit_transform(features['team2'])
X = features[['venue', 'toss_decision', 'team1', 'team2']]
y = matches_df['team1_won']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Q8: Logistic regression model accuracy: {accuracy:.2f}")
pd.DataFrame({
    'Feature': ['venue', 'toss_decision', 'team1', 'team2'],
    'Coefficient': model.coef_[0]
}).to_csv('logistic_regression_coefficients.csv')
print("Q8: Logistic regression coefficients saved to 'logistic_regression_coefficients.csv'.")
# Plot feature importance
plt.figure(figsize=(8, 6))
pd.Series(model.coef_[0], index=['venue', 'toss_decision', 'team1', 'team2']).plot(kind='bar', color='teal')
plt.title('Logistic Regression Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png')
plt.close()
print("Q8: Plot saved to 'logistic_regression_coefficients.png'.")
```
-------------------

```python


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the datasets
matches_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/matches_data.csv')
deliveries_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/deliveries_data.csv')

# Question 1: Batting strike rate of each batsman (minimum 500 runs)
batsman_stats = deliveries_df.groupby('batsman').agg({
    'batsman_runs': 'sum',
    'ball': 'count'  # Count legal balls (excluding wides)
})
batsman_stats['strike_rate'] = (batsman_stats['batsman_runs'] / batsman_stats['ball'] * 100).round(2)
batsman_stats = batsman_stats[batsman_stats['batsman_runs'] >= 500][['strike_rate']].sort_values('strike_rate', ascending=False)
print("Q1: Batting strike rates (min 500 runs):\n", batsman_stats.head())
batsman_stats.to_csv('batsman_strike_rates.csv')
print("Q1: Batting strike rates saved to 'batsman_strike_rates.csv'.")

# Question 2: Team with highest run rate in powerplay overs (1–6)
powerplay_df = deliveries_df[deliveries_df['over'].between(1, 6)]
team_powerplay = powerplay_df.groupby('batting_team').agg({
    'total_runs': 'sum',
    'over': 'count'  # Count balls
})
team_powerplay['overs'] = team_powerplay['over'] / 6  # Convert balls to overs
team_powerplay['run_rate'] = (team_powerplay['total_runs'] / team_powerplay['overs']).round(2)
top_team = team_powerplay['run_rate'].idxmax()
top_run_rate = team_powerplay['run_rate'].max()
print(f"Q2: Team with highest powerplay run rate is {top_team} with {top_run_rate:.2f} runs per over.")
team_powerplay[['run_rate']].to_csv('powerplay_run_rates.csv')
print("Q2: Powerplay run rates saved to 'powerplay_run_rates.csv'.")

# Question 3: Impact of toss win on match outcomes
matches_df['toss_win_match_win'] = matches_df['toss_winner'] == matches_df['winner']
toss_impact = matches_df.groupby('toss_winner')['toss_win_match_win'].mean().round(2) * 100
print("Q3: Win percentage when winning toss:\ poved", toss_impact)
toss_impact.to_csv('toss_impact.csv')
print("Q3: Toss impact data saved to 'toss_impact.csv'.")

# Question 4: Bowler with best economy rate in death overs (16–20, min 50 overs)
death_overs_df = deliveries_df[deliveries_df['over'].between(16, 20)]
bowler_stats = death_overs_df.groupby('bowler').agg({
    'total_runs': 'sum',
    'over': 'count'
})
bowler_stats['overs'] = bowler_stats['over'] / 6
bowler_stats['economy_rate'] = (bowler_stats['total_runs'] / bowler_stats['overs']).round(2)
bowler_stats = bowler_stats[bowler_stats['overs'] >= 50][['economy_rate']].sort_values('economy_rate')
top_bowler = bowler_stats['economy_rate'].idxmin()
top_economy = bowler_stats['economy_rate'].min()
print(f"Q4: Bowler with best death overs economy is {top_bowler} with {top_economy:.2f} runs per over.")
bowler_stats.to_csv('death_overs_economy.csv')
print("Q4: Death overs economy rates saved to 'death_overs_economy.csv'.")

# Question 5: Impact of batting first vs. second on match outcomes
matches_df['batting_first'] = matches_df['toss_decision'].map({'bat': True, 'field': False})
matches_df['batting_first_won'] = matches_df.apply(
    lambda x: x['winner'] == x['team1'] if x['batting_first'] else x['winner'] == x['team2'], axis=1
)
batting_impact = matches_df.groupby('season')['batting_first_won'].mean().round(2) * 100
print("Q5: Win percentage when batting first:\n", batting_impact)
batting_impact.to_csv('batting_first_impact.csv')
print("Q5: Batting first impact saved to 'batting_first_impact.csv'.")

# Question 6: Batsman with highest average against a specific bowler (min 30 balls)
batsman_bowler = deliveries_df.groupby(['batsman', 'bowler']).agg({
    'batsman_runs': 'sum',
    'ball': 'count',
    'player_dismissed': lambda x: x.notna().sum()
})
batsman_bowler = batsman_bowler[batsman_bowler['ball'] >= 30]
batsman_bowler['batting_avg'] = (batsman_bowler['batsman_runs'] / batsman_bowler['player_dismissed'].replace(0, 1)).round(2)
top_batsman_bowler = batsman_bowler['batting_avg'].idxmax()
top_avg = batsman_bowler['batting_avg'].max()
print(f"Q6: Batsman with highest average against a bowler is {top_batsman_bowler[0]} against {top_batsman_bowler[1]} with {top_avg:.2f}.")
batsman_bowler[['batting_avg']].to_csv('batsman_bowler_avg.csv')
print("Q6: Batsman-bowler averages saved to 'batsman_bowler_avg.csv'.")

# Question 7: Dismissal types by batting team and season
deliveries_with_season = deliveries_df.merge(matches_df[['id', 'season']], left_on='match_id', right_on='id')
dismissal_types = deliveries_with_season.groupby(['season', 'batting_team', 'dismissal_kind']).size().unstack(fill_value=0)
print("Q7: Dismissal types by team and season:\n", dismissal_types.head())
dismissal_types.to_csv('dismissal_types.csv')
print("Q7: Dismissal types saved to 'dismissal_types.csv'.")

# Question 8: Logistic regression to predict team winning based on venue and toss decision
matches_df['team1_won'] = (matches_df['winner'] == matches_df['team1']).astype(int)
features = matches_df[['venue', 'toss_decision', 'team1', 'team2']]
le_venue = LabelEncoder()
le_toss = LabelEncoder()
le_team1 = LabelEncoder()
le_team2 = LabelEncoder()
features['venue'] = le_venue.fit_transform(features['venue'])
features['toss_decision'] = le_toss.fit_transform(features['toss_decision'])
features['team1'] = le_team1.fit_transform(features['team1'])
features['team2'] = le_team2.fit_transform(features['team2'])
X = features[['venue', 'toss_decision', 'team1', 'team2']]
y = matches_df['team1_won']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Q8: Logistic regression model accuracy: {accuracy:.2f}")
pd.DataFrame({
    'Feature': ['venue', 'toss_decision', 'team1', 'team2'],
    'Coefficient': model.coef_[0]
}).to_csv('logistic_regression_coefficients.csv')
print("Q8: Logistic regression coefficients saved to 'logistic_regression_coefficients.csv'.")

```

-------------

```python


import pandas as pd

# Load the datasets
matches_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/matches_data.csv')
deliveries_df = pd.read_csv('https://raw.githubusercontent.com/Pankaj-Str/IPL-Data-Analysis-Case-Study/refs/heads/main/dataset/deliveries_data.csv')

# Question 1: Which team has won the most matches in the IPL?
team_wins = matches_df['winner'].value_counts()
most_wins_team = team_wins.idxmax()
most_wins_count = team_wins.max()
print(f"Q1: The team with the most IPL match wins is {most_wins_team} with {most_wins_count} wins.")
team_wins.to_csv('team_wins.csv')
print("Q1: Team wins data saved to 'team_wins.csv'.")

# Question 2: What is the average win margin (by runs and wickets) for matches?
avg_win_by_runs = matches_df[matches_df['win_by_runs'] > 0]['win_by_runs'].mean()
avg_win_by_wickets = matches_df[matches_df['win_by_wickets'] > 0]['win_by_wickets'].mean()
print(f"Q2: Average win margin by runs: {avg_win_by_runs:.2f}, by wickets: {avg_win_by_wickets:.2f}")
pd.DataFrame({
    'Metric': ['Average Win by Runs', 'Average Win by Wickets'],
    'Value': [avg_win_by_runs, avg_win_by_wickets]
}).to_csv('win_margins.csv', index=False)
print("Q2: Win margins data saved to 'win_margins.csv'.")

# Question 3: Which player has won the most "Player of the Match" awards?
player_of_match = matches_df['player_of_match'].value_counts()
top_player = player_of_match.idxmax()
top_player_count = player_of_match.max()
print(f"Q3: The player with the most Player of the Match awards is {top_player} with {top_player_count} awards.")
player_of_match.to_csv('player_of_match.csv')
print("Q3: Player of the Match data saved to 'player_of_match.csv'.")

# Question 4: How many matches were played in each season?
matches_per_season = matches_df['season'].value_counts().sort_index()
print(f"Q4: Matches played per season:\n{matches_per_season}")
matches_per_season.to_csv('matches_per_season.csv')
print("Q4: Matches per season data saved to 'matches_per_season.csv'.")

# Question 5: Which venue hosted the most IPL matches?
venue_counts = matches_df['venue'].value_counts()
top_venue = venue_counts.idxmax()
top_venue_count = venue_counts.max()
print(f"Q5: The venue with the most IPL matches is {top_venue} with {top_venue_count} matches.")
venue_counts.to_csv('venue_counts.csv')
print("Q5: Venue counts data saved to 'venue_counts.csv'.")

# Question 6: What is the total number of runs scored by each team in the IPL?
team_runs = deliveries_df.groupby('batting_team')['total_runs'].sum()
print(f"Q6: Total runs scored by each team:\n{team_runs}")
team_runs.to_csv('team_runs.csv')
print("Q6: Team runs data saved to 'team_runs.csv'.")

# Question 7: Which bowler has taken the most wickets in the IPL?
wickets = deliveries_df[deliveries_df['player_dismissed'].notna() & (deliveries_df['dismissal_kind'] != 'run out')]
bowler_wickets = wickets.groupby('bowler')['player_dismissed'].count()
top_bowler = bowler_wickets.idxmax()
top_bowler_wickets = bowler_wickets.max()
print(f"Q7: The bowler with the most wickets is {top_bowler} with {top_bowler_wickets} wickets.")
bowler_wickets.to_csv('bowler_wickets.csv')
print("Q7: Bowler wickets data saved to 'bowler_wickets.csv'.")

# Question 8: What is the distribution of toss decisions (bat/field) across seasons?
toss_decisions = matches_df.groupby(['season', 'toss_decision']).size().unstack(fill_value=0)
print(f"Q8: Toss decisions distribution across seasons:\n{toss_decisions}")
toss_decisions.to_csv('toss_decisions.csv')
print("Q8: Toss decisions data saved to 'toss_decisions.csv'.")

````