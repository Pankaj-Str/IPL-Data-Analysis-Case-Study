### 🏏 IPL Data Analysis Case Study

**Description:**

This repository presents an in-depth data analysis case study on the Indian Premier League (IPL), focusing on uncovering patterns, insights, and trends using Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn. The project explores team performances, player statistics, toss decisions, match outcomes, and other key aspects of the IPL dataset. This case study is ideal for data enthusiasts looking to enhance their data wrangling, visualization, and storytelling skills through real-world sports data.

**Key Features:**

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Visualizations using Seaborn & Matplotlib
* Team-wise and Player-wise Performance Analysis
* Toss Impact & Match Outcome Study
* Jupyter Notebook for Interactive Exploration

---

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

