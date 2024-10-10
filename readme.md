# NFL Fantasy Football Machine Learning Setup

This guide provides a step-by-step process to set up an environment for collecting both historical NFL data and current DraftKings data. You’ll use this data to develop machine learning models aimed at predicting and creating winning lineups for NFL fantasy football.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Set Up Your Python Environment](#step-1-set-up-your-python-environment)
- [Step 2: Install and Set Up the DraftKings Client](#step-2-install-and-set-up-the-draftkings-client)
- [Step 3: Retrieve Historical NFL Data with nfl_data_py](#step-3-retrieve-historical-nfl-data-with-nfl_data_py)
- [Step 4: Use nfl_data_py for Additional Historical Data](#step-4-use-nfl_data_py-for-additional-historical-data)
- [Step 5: Optional: Use NFLVerse for Advanced Analysis](#step-5-optional-use-nflverse-for-advanced-analysis)
- [Step 6: Combine Data for Machine Learning](#step-6-combine-data-for-machine-learning)
- [Step 7: Automate Data Fetching (Optional)](#step-7-automate-data-fetching-optional)
- [Key Repositories and Resources](#key-repositories-and-resources)
- [Final Thoughts](#final-thoughts)
- [Acknowledgments](#acknowledgments)

---

## Prerequisites

- **Python 3.7+**: Ensure you have Python installed. Download it from [python.org](https://www.python.org/).
- **pip**: Python package manager. Usually installed with Python.

---

## Step 1: Set Up Your Python Environment

Install the necessary libraries using pip:

```bash
pip install pandas requests draft-kings nfl_data_py

Note: We are using draft-kings instead of draftkings_client due to package naming.

Step 2: Install and Set Up the DraftKings Client

The DraftKings Client allows you to retrieve current contest and player salary data from DraftKings.

	1.	Install the package:

pip install draft-kings

	2.	Use the DraftKings API to get contests and player salaries:

from draft_kings import Sport, Client

client = Client()

# Fetch NFL contests from DraftKings
contests = client.contests(sport=Sport.NFL)
print(contests)

# Extract the draft group ID from the contest data
draft_group_id = contests['draftGroups'][0]['draftGroupId'] # Adjust indexing as needed

# Get draftable players for the specific draft group
players = client.available_players(draft_group_id=draft_group_id)
print(players)

	•	Note: Ensure you extract the correct draftGroupId from the contests data, as it’s essential for fetching player information.

Step 3: Retrieve Historical NFL Data with nfl_data_py

The nfldfs package is outdated. Instead, use nfl_data_py to access historical DFS data.

	1.	Install the package:

pip install nfl_data_py

	2.	Fetch historical DraftKings salary and points data:

import nfl_data_py as nfl

# Fetch historical DraftKings data for the 2019 season
dk_data = nfl.import_draftkings_weekly_data([2019])
print(dk_data.head())

# Export data to CSV for later use
dk_data.to_csv("nfl_draftkings_data.csv", index=False)

	•	Advantages:
	•	nfl_data_py is actively maintained.
	•	Provides reliable access to historical DFS data.

Step 4: Use nfl_data_py for Additional Historical Data

Access a broader range of historical NFL data, including play-by-play and player stats.

	1.	Fetch player stats for 2020:

import nfl_data_py as nfl

# Fetch player stats for 2020
player_stats = nfl.import_player_stats([2020])
print(player_stats.head())

# Fetch roster data for 2020
roster_data = nfl.import_rosters([2020])
print(roster_data.head())

	•	Note: Play-by-play data (import_pbp_data) can be large (multiple GBs). Ensure your system has enough memory and storage.

Step 5: Optional: Use NFLVerse for Advanced Analysis

If you’re comfortable with R, you can explore NFLVerse R packages for in-depth NFL analysis. However, most of the NFLVerse data is accessible via Python using nfl_data_py.

	•	NFLVerse Official Website: https://nflverse.com/
	•	NFLVerse GitHub Organization: https://github.com/nflverse

Step 6: Combine Data for Machine Learning

With both historical and current data collected, proceed to prepare your dataset for machine learning.

1. Data Preprocessing

	•	Merge Datasets:
	•	Align and merge DFS data with player stats and roster data using common keys like player IDs and game IDs.
	•	Handle Missing Values:
	•	Fill or remove missing values to prevent errors during modeling.

2. Feature Engineering

	•	Create New Features:
	•	Develop features such as:
	•	Recent performance trends.
	•	Opponent defensive rankings.
	•	Weather conditions.
	•	Salary-to-performance ratios.
	•	Encode Categorical Variables:
	•	Convert categorical data (e.g., player positions) into numerical formats using one-hot encoding or label encoding.

3. Modeling

	•	Select Appropriate Models:
	•	Start with models like Random Forest, Gradient Boosting (e.g., XGBoost), or neural networks.
	•	Cross-Validation:
	•	Use techniques like k-fold cross-validation to evaluate model performance.

4. Optimization

	•	Lineup Construction:
	•	Apply optimization algorithms (e.g., linear programming) to select the best lineup under salary cap constraints.
	•	Constraints:
	•	Incorporate contest rules, such as position requirements and team stacking limitations.

Step 7: Automate Data Fetching (Optional)

Automate the data retrieval by scheduling Python scripts to run regularly.

Scheduling Tools

	•	Cron Jobs (Unix/Linux):
Schedule scripts using cron:

# Example: Run every Thursday at 6 AM
0 6 * * 4 /usr/bin/python3 /path/to/your_script.py


	•	Task Scheduler (Windows):
Use the built-in Task Scheduler to run scripts at specified times.

Cloud Services

	•	AWS Lambda:
Deploy serverless functions that run code in response to events.
	•	Azure Functions:
Similar to AWS Lambda, for automating tasks in the Azure cloud.

Key Repositories and Resources

	•	DraftKings Client:
	•	GitHub Repository: https://github.com/jaebradley/draftkings
	•	nfl_data_py:
	•	GitHub Repository: https://github.com/derek-adair/nfl_data_py
	•	NFLVerse:
	•	Official Website: https://nflverse.com/
	•	GitHub Organization: https://github.com/nflverse
	•	Machine Learning Examples and Datasets:
	•	Kaggle NFL Datasets: https://www.kaggle.com/c/nfl-big-data-bowl-2021

Final Thoughts

By following this guide, you will have a functional environment capable of fetching both historical and current NFL data for analysis. Here are some additional tips to ensure success:

Environment Management

	•	Virtual Environments:
Use venv or conda to manage your project’s dependencies and isolate them from your global Python environment.

# Using venv
python3 -m venv venv
source venv/bin/activate

# Using conda
conda create -n nfl_ml_env python=3.8
conda activate nfl_ml_env



Data Storage

	•	Efficient Formats:
For large datasets, consider using efficient storage formats like Parquet instead of CSV.

dk_data.to_parquet("nfl_draftkings_data.parquet", index=False)



Version Control

	•	Git:
Use Git for version control to track changes in your codebase.

git init
git add .
git commit -m "Initial commit"


	•	GitHub:
Host your repository on GitHub for collaboration and backup.

git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin master



Acknowledgments

	•	DraftKings Client by jaebradley
	•	nfl_data_py by derek-adair
	•	NFLVerse community at nflverse.com and GitHub

This README was generated to assist in setting up an environment for NFL fantasy football machine learning projects, focusing on data retrieval and preparation for predictive modeling.

```
