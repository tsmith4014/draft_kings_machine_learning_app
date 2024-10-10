import requests  # For making HTTP requests
import pandas as pd  # For data manipulation and analysis
import os  # For interacting with the operating system
from sklearn.linear_model import LinearRegression  # For linear regression
from sklearn.preprocessing import PolynomialFeatures  # For polynomial regression
from sklearn.tree import DecisionTreeRegressor  # For decision tree regression
from sklearn.ensemble import RandomForestRegressor  # For random forest regression
from flask import Flask, request, jsonify, render_template, url_for
import json  # For working with JSON data
import re  # For regular expressions
from bs4 import BeautifulSoup  # For parsing HTML content
import pulp  # For linear programming optimization
import unicodedata  # For Unicode character database

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Define constants for URLs and file paths
BASE_URL = "https://www.draftkings.com/lineup/getavailableplayerscsv"
CSV_SAVE_PATH = "explore_output/draft_players.csv"
STATIC_DIR = "static"
JSON_DATA_DIR = STATIC_DIR  # Directory to save JSON data

# Ensure output directories exist
os.makedirs("explore_output", exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Function to extract contest IDs from the user-provided URL
def extract_contest_ids(contest_url):
    try:
        response = requests.get(contest_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Search for the script tag containing 'draftGroupId'
        script_tags = soup.find_all('script')
        for script in script_tags:
            if 'draftGroupId' in script.text:
                match = re.search(r'draftGroupId":(\d+)', script.text)
                if match:
                    draft_group_id = match.group(1)
                    contest_type_id = 21  # Assuming contest_type_id is always 21
                    return contest_type_id, draft_group_id

        # If draftGroupId is not found
        raise ValueError("Unable to extract draft group ID from the page.")
    except Exception as e:
        print(f"Error extracting contest IDs: {e}")
        return None, None

# Function to download the player CSV file
def download_player_csv(contest_type_id, draft_group_id):
    try:
        url = f"{BASE_URL}?contestTypeId={contest_type_id}&draftGroupId={draft_group_id}"
        response = requests.get(url)
        response.raise_for_status()

        # Save the CSV content to a file
        with open(CSV_SAVE_PATH, 'w') as csv_file:
            csv_file.write(response.text)
        print(f"CSV downloaded successfully to {CSV_SAVE_PATH}")
    except requests.RequestException as e:
        print(f"Error fetching CSV: {e}")

# Function to parse the downloaded CSV file
def parse_player_csv():
    try:
        player_df = pd.read_csv(CSV_SAVE_PATH)

        # Filter out players in games that are in progress
        player_df = player_df[~player_df['Game Info'].str.contains('In Progress', na=False)]
        print("Player data loaded successfully.")
        print(f"Number of players loaded: {len(player_df)}")
        return player_df
    except FileNotFoundError:
        print("CSV file not found. Please download it first.")
    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
    except Exception as e:
        print(f"Unexpected error while reading CSV: {e}")

# Function to create visualizations and save JSON data
def create_visualizations(player_df, model_type='linear'):
    # Make a copy to avoid modifying the original DataFrame
    player_df = player_df.copy()
    if 'Salary' in player_df.columns and 'AvgPointsPerGame' in player_df.columns and 'Name' in player_df.columns:
        try:
            # Prepare data by dropping rows with missing values
            player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])
            if len(player_df) == 0:
                raise ValueError("No players available after filtering.")

            # Ensure columns are numeric
            player_df['Salary'] = pd.to_numeric(player_df['Salary'].replace(',', '', regex=True), errors='coerce')
            player_df['AvgPointsPerGame'] = pd.to_numeric(player_df['AvgPointsPerGame'], errors='coerce')
            player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])

            # Prepare data for regression
            X = player_df[['Salary']].values
            y = player_df['AvgPointsPerGame'].values
            player_names = player_df['Name'].values

            # Select and fit the regression model based on model_type
            if model_type == 'linear':
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
            elif model_type == 'polynomial':
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
            elif model_type == 'decision_tree':
                model = DecisionTreeRegressor(max_depth=4)
                model.fit(X, y)
                y_pred = model.predict(X)
            elif model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                y_pred = model.predict(X)
            else:
                raise ValueError("Invalid model_type specified.")

            # Save data to JSON for Plotly
            data = {
                "Salary": player_df['Salary'].tolist(),
                "AvgPointsPerGame": y.tolist(),
                "Name": player_names.tolist(),
                "RegressionLine": y_pred.tolist()
            }

            # Define JSON file path based on model type
            json_file_path = os.path.join(JSON_DATA_DIR, f"points_vs_salary_data_{model_type}.json")

            with open(json_file_path, 'w') as json_file:
                json.dump(data, json_file)
            print(f"Data saved to {json_file_path}")

            # Add regression predictions to player data
            player_df[f'{model_type}_PredictedPoints'] = y_pred

            return player_df  # Return the updated DataFrame

        except Exception as e:
            print(f"Error creating visualization: {e}")
    else:
        print("Required columns not found in player data.")

    return player_df  # Return the DataFrame even if there's an error

# Function to normalize player names for matching
def normalize_name(name):
    # Remove accents and special characters
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    # Remove non-alphanumeric characters except spaces
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    # Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name)
    # Convert to lowercase
    return name.strip().lower()

# Function to calculate Value Over Replacement Player (VORP)
def calculate_vorp(player_df):
    # Make a copy to avoid modifying the original DataFrame
    player_df = player_df.copy()
    # Calculate the average points per position
    avg_points_per_position = player_df.groupby('Position')['AvgPointsPerGame'].mean()
    # Map the average points to each player
    player_df['ReplacementLevel'] = player_df['Position'].map(avg_points_per_position)
    # Calculate VORP
    player_df['VORP'] = player_df['AvgPointsPerGame'] - player_df['ReplacementLevel']
    return player_df

# Function to find optimal lineups using linear programming
def find_optimal_lineups(player_df, exclude_players, optimization_method='standard', model_type='linear'):
    # Make a copy to avoid modifying the original DataFrame
    player_df = player_df.copy()

    # Normalize player names for matching
    player_df['Normalized_Name'] = player_df['Name'].apply(normalize_name)

    # Normalize and process exclude_players input
    exclude_players = [normalize_name(name) for name in exclude_players.split(",") if name.strip()]
    print(f"Excluding players: {exclude_players}")

    # Check for unmatched player names
    unmatched_players = [name for name in exclude_players if name not in player_df['Normalized_Name'].values]
    if unmatched_players:
        print(f"The following players were not found and could not be excluded: {unmatched_players}")

    # Ensure columns are numeric
    player_df['Salary'] = pd.to_numeric(player_df['Salary'].replace(',', '', regex=True), errors='coerce')
    player_df['AvgPointsPerGame'] = pd.to_numeric(player_df['AvgPointsPerGame'], errors='coerce')
    player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])

    # Exclude specified players
    player_df = player_df[~player_df['Normalized_Name'].isin(exclude_players)]

    # Create a unique ID for each player
    player_df['ID'] = player_df.index

    # Prepare the objective based on the optimization method
    if optimization_method == 'standard':
        # Use AvgPointsPerGame
        player_df['ObjectivePoints'] = player_df['AvgPointsPerGame']
    elif optimization_method == 'regression':
        # Use regression model predictions
        pred_column = f'{model_type}_PredictedPoints'
        if pred_column not in player_df.columns:
            print(f"Regression predictions not found for model {model_type}. Using AvgPointsPerGame instead.")
            player_df['ObjectivePoints'] = player_df['AvgPointsPerGame']
        else:
            player_df['ObjectivePoints'] = player_df[pred_column]
    elif optimization_method == 'vorp':
        # Use VORP
        player_df = calculate_vorp(player_df)
        player_df['ObjectivePoints'] = player_df['VORP']
    else:
        print(f"Invalid optimization method {optimization_method}. Using standard method.")
        player_df['ObjectivePoints'] = player_df['AvgPointsPerGame']

    # Display counts of available players by position
    position_counts = player_df['Position'].value_counts()
    print("Available players by position:")
    print(position_counts)

    # Initialize list to store lineups
    lineups = []

    # Generate multiple optimal lineups
    for lineup_number in range(3):
        # Define the optimization problem
        prob = pulp.LpProblem(f"DraftKings_Lineup_Optimization_{lineup_number+1}", pulp.LpMaximize)

        # Create binary variables for each player
        player_vars = pulp.LpVariable.dicts("Player", player_df['ID'], cat='Binary')

        # Objective function: Maximize total objective points
        prob += pulp.lpSum([player_vars[player_id] * player_df.loc[player_id, 'ObjectivePoints'] for player_id in player_df['ID']])

        # Salary cap constraint
        prob += pulp.lpSum([player_vars[player_id] * player_df.loc[player_id, 'Salary'] for player_id in player_df['ID']]) <= 50000

        # Total players constraint
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df['ID']]) == 9

        # Position constraints
        # QB constraint
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'QB']['ID']]) == 1

        # RB constraints (2 to 3)
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'RB']['ID']]) >= 2
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'RB']['ID']]) <= 3

        # WR constraints (3 to 4)
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'WR']['ID']]) >= 3
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'WR']['ID']]) <= 4

        # TE constraints (1 to 2)
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'TE']['ID']]) >= 1
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'TE']['ID']]) <= 2

        # FLEX position constraint (total RB/WR/TE should be 7)
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'].isin(['RB', 'WR', 'TE'])]['ID']]) == 7

        # DST constraint
        prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'DST']['ID']]) == 1

        # Ensure diversity among generated lineups
        if lineups:
            for prev_lineup in lineups:
                prob += pulp.lpSum([player_vars[player['ID']] for player in prev_lineup['lineup']]) <= 8  # At least one player different

        # Solve the optimization problem
        prob.solve()

        # Check if an optimal solution was found
        if pulp.LpStatus[prob.status] != 'Optimal':
            print(f"No optimal solution found for lineup {lineup_number+1}.")
            continue

        # Extract the lineup from the solution
        lineup = []
        for player_id in player_df['ID']:
            if player_vars[player_id].varValue == 1:
                player_info = player_df.loc[player_id].to_dict()
                lineup.append(player_info)

        # Verify salary cap
        total_salary = sum(player['Salary'] for player in lineup)
        if total_salary > 50000:
            print(f"Lineup {lineup_number+1} exceeds the salary cap.")
            continue

        # Calculate total projected points
        total_points = sum(player['ObjectivePoints'] for player in lineup)

        # Log the lineup details
        lineup_names = [player['Name'] for player in lineup]
        print(f"Lineup {lineup_number+1} players: {lineup_names}")
        print(f"Total salary for lineup {lineup_number+1}: {total_salary}")
        print(f"Total projected points for lineup {lineup_number+1}: {total_points}")
        print(f"Lineup {lineup_number+1} added.")

        # Collect solver statistics
        solver_status = pulp.LpStatus[prob.status]

        # Add lineup and stats to the lineups list
        lineups.append({
            'lineup': lineup,
            'total_salary': total_salary,
            'total_points': total_points,
            'solver_status': solver_status
        })

    print(f"Total lineups generated: {len(lineups)}")
    return lineups  # Return the list of generated lineups

# Flask route to get player details as JSON
@app.route('/api/players', methods=['GET'])
def get_players():
    try:
        player_df = parse_player_csv()
        if player_df is not None:
            return jsonify(player_df.to_dict(orient='records'))
        return jsonify({"error": "Player data not available."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask route to render the visualization page
@app.route('/visualization', methods=['GET'])
def render_visualization():
    try:
        return render_template('visualization.html')
    except Exception as e:
        return f"An error occurred: {e}", 500

# Flask route to get optimal lineups as JSON
@app.route('/api/optimal_lineups', methods=['GET'])
def get_optimal_lineups():
    try:
        player_df = parse_player_csv()
        exclude_players = request.args.get('exclude_players', '')
        optimization_method = request.args.get('optimization_method', 'standard')
        model_type = request.args.get('model_type', 'linear')
        print(f"Received exclude_players: {exclude_players}")
        print(f"Optimization method: {optimization_method}")
        print(f"Model type: {model_type}")
        if player_df is not None and not player_df.empty:
            # Generate visualizations if regression method is used
            if optimization_method == 'regression':
                player_df = create_visualizations(player_df, model_type=model_type)  # Update player_df with predictions
            lineups = find_optimal_lineups(player_df, exclude_players, optimization_method, model_type)
            return jsonify({'lineups': lineups})
        return jsonify({"error": "Player data not available or insufficient data."}), 404
    except Exception as e:
        print(f"Exception in get_optimal_lineups: {e}")
        return jsonify({"error": str(e)}), 500

# Flask route to process the contest link submitted by the user
@app.route('/api/process_contest_link', methods=['POST'])
def process_contest_link():
    try:
        contest_url = request.json.get('contest_url')
        if contest_url:
            contest_type_id, draft_group_id = extract_contest_ids(contest_url)
            if contest_type_id and draft_group_id:
                download_player_csv(contest_type_id, draft_group_id)
                # After downloading, create visualizations using different models
                player_df = parse_player_csv()
                if player_df is not None and not player_df.empty:
                    # Generate visualizations for all supported models
                    for model_type in ['linear', 'polynomial', 'decision_tree', 'random_forest']:
                        create_visualizations(player_df, model_type=model_type)
                return jsonify({"message": "Contest data successfully fetched."}), 200
            else:
                return jsonify({"error": "Invalid contest URL format."}), 400
        else:
            return jsonify({"error": "Contest URL not provided."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask app in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5001)



# import requests  # For making HTTP requests
# import pandas as pd  # For data manipulation and analysis
# import os  # For interacting with the operating system
# from sklearn.linear_model import LinearRegression  # For linear regression
# from sklearn.preprocessing import PolynomialFeatures  # For polynomial regression
# from sklearn.tree import DecisionTreeRegressor  # For decision tree regression
# from sklearn.ensemble import RandomForestRegressor  # For random forest regression
# from flask import Flask, request, jsonify, render_template, url_for
# import json  # For working with JSON data
# import re  # For regular expressions
# from bs4 import BeautifulSoup  # For parsing HTML content
# import pulp  # For linear programming optimization
# import unicodedata  # For Unicode character database

# # Initialize Flask app
# app = Flask(__name__, template_folder='templates')

# # Define constants for URLs and file paths
# BASE_URL = "https://www.draftkings.com/lineup/getavailableplayerscsv"
# CSV_SAVE_PATH = "explore_output/draft_players.csv"
# STATIC_DIR = "static"
# JSON_DATA_DIR = STATIC_DIR  # Directory to save JSON data

# # Ensure output directories exist
# os.makedirs("explore_output", exist_ok=True)
# os.makedirs(STATIC_DIR, exist_ok=True)

# # Function to extract contest IDs from the user-provided URL
# def extract_contest_ids(contest_url):
#     try:
#         response = requests.get(contest_url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # Search for the script tag containing 'draftGroupId'
#         script_tags = soup.find_all('script')
#         for script in script_tags:
#             if 'draftGroupId' in script.text:
#                 match = re.search(r'draftGroupId":(\d+)', script.text)
#                 if match:
#                     draft_group_id = match.group(1)
#                     contest_type_id = 21  # Assuming contest_type_id is always 21
#                     return contest_type_id, draft_group_id

#         # If draftGroupId is not found
#         raise ValueError("Unable to extract draft group ID from the page.")
#     except Exception as e:
#         print(f"Error extracting contest IDs: {e}")
#         return None, None

# # Function to download the player CSV file
# def download_player_csv(contest_type_id, draft_group_id):
#     try:
#         url = f"{BASE_URL}?contestTypeId={contest_type_id}&draftGroupId={draft_group_id}"
#         response = requests.get(url)
#         response.raise_for_status()

#         # Save the CSV content to a file
#         with open(CSV_SAVE_PATH, 'w') as csv_file:
#             csv_file.write(response.text)
#         print(f"CSV downloaded successfully to {CSV_SAVE_PATH}")
#     except requests.RequestException as e:
#         print(f"Error fetching CSV: {e}")

# # Function to parse the downloaded CSV file
# def parse_player_csv():
#     try:
#         player_df = pd.read_csv(CSV_SAVE_PATH)

#         # Filter out players in games that are in progress
#         player_df = player_df[~player_df['Game Info'].str.contains('In Progress', na=False)]
#         print("Player data loaded successfully.")
#         print(f"Number of players loaded: {len(player_df)}")
#         return player_df
#     except FileNotFoundError:
#         print("CSV file not found. Please download it first.")
#     except pd.errors.EmptyDataError:
#         print("CSV file is empty.")
#     except Exception as e:
#         print(f"Unexpected error while reading CSV: {e}")

# # Function to create visualizations and save JSON data
# def create_visualizations(player_df, model_type='linear'):
#     # Make a copy to avoid modifying the original DataFrame
#     player_df = player_df.copy()
#     if 'Salary' in player_df.columns and 'AvgPointsPerGame' in player_df.columns and 'Name' in player_df.columns:
#         try:
#             # Prepare data by dropping rows with missing values
#             player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])
#             if len(player_df) == 0:
#                 raise ValueError("No players available after filtering.")

#             # Ensure columns are numeric
#             player_df['Salary'] = pd.to_numeric(player_df['Salary'].replace(',', '', regex=True), errors='coerce')
#             player_df['AvgPointsPerGame'] = pd.to_numeric(player_df['AvgPointsPerGame'], errors='coerce')
#             player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])

#             # Prepare data for regression
#             X = player_df[['Salary']].values
#             y = player_df['AvgPointsPerGame'].values
#             player_names = player_df['Name'].values

#             # Select and fit the regression model based on model_type
#             if model_type == 'linear':
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 y_pred = model.predict(X)
#             elif model_type == 'polynomial':
#                 poly = PolynomialFeatures(degree=2)
#                 X_poly = poly.fit_transform(X)
#                 model = LinearRegression()
#                 model.fit(X_poly, y)
#                 y_pred = model.predict(X_poly)
#             elif model_type == 'decision_tree':
#                 model = DecisionTreeRegressor(max_depth=4)
#                 model.fit(X, y)
#                 y_pred = model.predict(X)
#             elif model_type == 'random_forest':
#                 model = RandomForestRegressor(n_estimators=100, random_state=42)
#                 model.fit(X, y)
#                 y_pred = model.predict(X)
#             else:
#                 raise ValueError("Invalid model_type specified.")

#             # Save data to JSON for Plotly
#             data = {
#                 "Salary": X.flatten().tolist(),
#                 "AvgPointsPerGame": y.tolist(),
#                 "Name": player_names.tolist(),
#                 "RegressionLine": y_pred.tolist()
#             }

#             # Define JSON file path based on model type
#             json_file_path = os.path.join(JSON_DATA_DIR, f"points_vs_salary_data_{model_type}.json")

#             with open(json_file_path, 'w') as json_file:
#                 json.dump(data, json_file)
#             print(f"Data saved to {json_file_path}")

#             # Add regression predictions to player data
#             player_df['RegressionPoints'] = y_pred

#             return player_df

#         except Exception as e:
#             print(f"Error creating visualization: {e}")
#     else:
#         print("Required columns not found in player data.")

# # Function to normalize player names for matching
# def normalize_name(name):
#     # Remove accents and special characters
#     name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
#     # Remove non-alphanumeric characters except spaces
#     name = re.sub(r'[^a-zA-Z\s]', '', name)
#     # Replace multiple spaces with a single space
#     name = re.sub(r'\s+', ' ', name)
#     # Convert to lowercase
#     return name.strip().lower()

# # Function to find optimal lineups using linear programming
# def find_optimal_lineups(player_df, exclude_players):
#     # Make a copy to avoid modifying the original DataFrame
#     player_df = player_df.copy()

#     # Normalize player names for matching
#     player_df['Normalized_Name'] = player_df['Name'].apply(normalize_name)

#     # Normalize and process exclude_players input
#     exclude_players = [normalize_name(name) for name in exclude_players.split(",") if name.strip()]
#     print(f"Excluding players: {exclude_players}")

#     # Check for unmatched player names
#     unmatched_players = [name for name in exclude_players if name not in player_df['Normalized_Name'].values]
#     if unmatched_players:
#         print(f"The following players were not found and could not be excluded: {unmatched_players}")

#     # Ensure columns are numeric
#     player_df['Salary'] = pd.to_numeric(player_df['Salary'].replace(',', '', regex=True), errors='coerce')
#     player_df['AvgPointsPerGame'] = pd.to_numeric(player_df['AvgPointsPerGame'], errors='coerce')
#     player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])

#     # Exclude specified players
#     player_df = player_df[~player_df['Normalized_Name'].isin(exclude_players)]

#     # Create a unique ID for each player
#     player_df['ID'] = player_df.index

#     # Display counts of available players by position
#     position_counts = player_df['Position'].value_counts()
#     print("Available players by position:")
#     print(position_counts)

#     # Initialize list to store lineups
#     lineups = []

#     # Generate multiple optimal lineups
#     for lineup_number in range(3):
#         # Define the optimization problem
#         prob = pulp.LpProblem(f"DraftKings_Lineup_Optimization_{lineup_number+1}", pulp.LpMaximize)

#         # Create binary variables for each player
#         player_vars = pulp.LpVariable.dicts("Player", player_df['ID'], cat='Binary')

#         # Objective function: Maximize total average points per game
#         prob += pulp.lpSum([player_vars[player_id] * player_df.loc[player_id, 'AvgPointsPerGame'] for player_id in player_df['ID']])

#         # Salary cap constraint
#         prob += pulp.lpSum([player_vars[player_id] * player_df.loc[player_id, 'Salary'] for player_id in player_df['ID']]) <= 50000

#         # Total players constraint
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df['ID']]) == 9

#         # Position constraints
#         # QB constraint
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'QB']['ID']]) == 1

#         # RB constraints (2 to 3)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'RB']['ID']]) >= 2
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'RB']['ID']]) <= 3

#         # WR constraints (3 to 4)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'WR']['ID']]) >= 3
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'WR']['ID']]) <= 4

#         # TE constraints (1 to 2)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'TE']['ID']]) >= 1
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'TE']['ID']]) <= 2

#         # FLEX position constraint (total RB/WR/TE should be 7)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'].isin(['RB', 'WR', 'TE'])]['ID']]) == 7

#         # DST constraint
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'DST']['ID']]) == 1

#         # Ensure diversity among generated lineups
#         if lineups:
#             for prev_lineup in lineups:
#                 prob += pulp.lpSum([player_vars[player['ID']] for player in prev_lineup]) <= 8  # At least one player different

#         # Solve the optimization problem
#         prob.solve()

#         # Check if an optimal solution was found
#         if pulp.LpStatus[prob.status] != 'Optimal':
#             print(f"No optimal solution found for lineup {lineup_number+1}.")
#             continue

#         # Extract the lineup from the solution
#         lineup = []
#         for player_id in player_df['ID']:
#             if player_vars[player_id].varValue == 1:
#                 player_info = player_df.loc[player_id].to_dict()
#                 lineup.append(player_info)

#         # Verify salary cap
#         total_salary = sum(player['Salary'] for player in lineup)
#         if total_salary > 50000:
#             print(f"Lineup {lineup_number+1} exceeds the salary cap.")
#             continue

#         # Log the lineup details
#         lineup_names = [player['Name'] for player in lineup]
#         print(f"Lineup {lineup_number+1} players: {lineup_names}")
#         print(f"Total salary for lineup {lineup_number+1}: {total_salary}")
#         print(f"Lineup {lineup_number+1} added.")
#         lineups.append(lineup)

#     print(f"Total lineups generated: {len(lineups)}")
#     return lineups  # Return the list of generated lineups

# # Flask route to get player details as JSON
# @app.route('/api/players', methods=['GET'])
# def get_players():
#     try:
#         player_df = parse_player_csv()
#         if player_df is not None:
#             return jsonify(player_df.to_dict(orient='records'))
#         return jsonify({"error": "Player data not available."}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Flask route to render the visualization page
# @app.route('/visualization', methods=['GET'])
# def render_visualization():
#     try:
#         return render_template('visualization.html')
#     except Exception as e:
#         return f"An error occurred: {e}", 500

# # Flask route to get optimal lineups as JSON
# @app.route('/api/optimal_lineups', methods=['GET'])
# def get_optimal_lineups():
#     try:
#         player_df = parse_player_csv()
#         exclude_players = request.args.get('exclude_players', '')
#         print(f"Received exclude_players: {exclude_players}")
#         if player_df is not None and not player_df.empty:
#             lineups = find_optimal_lineups(player_df, exclude_players)
#             return jsonify(lineups)
#         return jsonify({"error": "Player data not available or insufficient data."}), 404
#     except Exception as e:
#         print(f"Exception in get_optimal_lineups: {e}")
#         return jsonify({"error": str(e)}), 500

# # Flask route to process the contest link submitted by the user
# @app.route('/api/process_contest_link', methods=['POST'])
# def process_contest_link():
#     try:
#         contest_url = request.json.get('contest_url')
#         if contest_url:
#             contest_type_id, draft_group_id = extract_contest_ids(contest_url)
#             if contest_type_id and draft_group_id:
#                 download_player_csv(contest_type_id, draft_group_id)
#                 # After downloading, create visualizations using different models
#                 player_df = parse_player_csv()
#                 if player_df is not None and not player_df.empty:
#                     # Generate visualizations for all supported models
#                     for model_type in ['linear', 'polynomial', 'decision_tree', 'random_forest']:
#                         create_visualizations(player_df, model_type=model_type)
#                 return jsonify({"message": "Contest data successfully fetched."}), 200
#             else:
#                 return jsonify({"error": "Invalid contest URL format."}), 400
#         else:
#             return jsonify({"error": "Contest URL not provided."}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     # Run the Flask app in debug mode for development
#     app.run(debug=True, host='0.0.0.0', port=5001)










#working backup below, above adding in more ML algos and techniques
# import requests  # For making HTTP requests
# import pandas as pd  # For data manipulation and analysis
# import os  # For interacting with the operating system
# from sklearn.linear_model import LinearRegression  # For performing linear regression
# from flask import Flask, request, jsonify, render_template, url_for
# import json  # For working with JSON data
# import re  # For regular expressions
# from bs4 import BeautifulSoup  # For parsing HTML content
# import pulp  # For linear programming optimization
# import unicodedata  # For Unicode character database

# # Initialize Flask app
# app = Flask(__name__, template_folder='templates')

# # Define constants for URLs and file paths
# BASE_URL = "https://www.draftkings.com/lineup/getavailableplayerscsv"
# CSV_SAVE_PATH = "explore_output/draft_players.csv"
# STATIC_DIR = "static"
# JSON_DATA_PATH = os.path.join(STATIC_DIR, "points_vs_salary_data.json")

# # Ensure output directories exist
# os.makedirs("explore_output", exist_ok=True)
# os.makedirs(STATIC_DIR, exist_ok=True)

# # Function to extract contest IDs from the user-provided URL
# def extract_contest_ids(contest_url):
#     try:
#         response = requests.get(contest_url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'html.parser')

#         # Search for the script tag containing 'draftGroupId'
#         script_tags = soup.find_all('script')
#         for script in script_tags:
#             if 'draftGroupId' in script.text:
#                 match = re.search(r'draftGroupId":(\d+)', script.text)
#                 if match:
#                     draft_group_id = match.group(1)
#                     contest_type_id = 21  # Assuming contest_type_id is always 21
#                     return contest_type_id, draft_group_id

#         # If draftGroupId is not found
#         raise ValueError("Unable to extract draft group ID from the page.")
#     except Exception as e:
#         print(f"Error extracting contest IDs: {e}")
#         return None, None

# # Function to download the player CSV file
# def download_player_csv(contest_type_id, draft_group_id):
#     try:
#         url = f"{BASE_URL}?contestTypeId={contest_type_id}&draftGroupId={draft_group_id}"
#         response = requests.get(url)
#         response.raise_for_status()

#         # Save the CSV content to a file
#         with open(CSV_SAVE_PATH, 'w') as csv_file:
#             csv_file.write(response.text)
#         print(f"CSV downloaded successfully to {CSV_SAVE_PATH}")
#     except requests.RequestException as e:
#         print(f"Error fetching CSV: {e}")

# # Function to parse the downloaded CSV file
# def parse_player_csv():
#     try:
#         player_df = pd.read_csv(CSV_SAVE_PATH)

#         # Filter out players in games that are in progress
#         player_df = player_df[~player_df['Game Info'].str.contains('In Progress', na=False)]
#         print("Player data loaded successfully.")
#         print(f"Number of players loaded: {len(player_df)}")
#         return player_df
#     except FileNotFoundError:
#         print("CSV file not found. Please download it first.")
#     except pd.errors.EmptyDataError:
#         print("CSV file is empty.")
#     except Exception as e:
#         print(f"Unexpected error while reading CSV: {e}")

# # Function to create visualizations and save JSON data
# def create_visualizations(player_df):
#     # Make a copy to avoid modifying the original DataFrame
#     player_df = player_df.copy()
#     if 'Salary' in player_df.columns and 'AvgPointsPerGame' in player_df.columns and 'Name' in player_df.columns:
#         try:
#             # Prepare data by dropping rows with missing values
#             player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])
#             if len(player_df) == 0:
#                 raise ValueError("No players available after filtering.")

#             # Ensure columns are numeric
#             player_df['Salary'] = pd.to_numeric(player_df['Salary'].replace(',', '', regex=True), errors='coerce')
#             player_df['AvgPointsPerGame'] = pd.to_numeric(player_df['AvgPointsPerGame'], errors='coerce')
#             player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])

#             # Prepare data for regression
#             X = player_df[['Salary']].values
#             y = player_df['AvgPointsPerGame'].values
#             player_names = player_df['Name'].values

#             # Create and fit the linear regression model
#             model = LinearRegression()
#             model.fit(X, y)
#             y_pred = model.predict(X)

#             # Save data to JSON for visualization
#             data = {
#                 "Salary": X.flatten().tolist(),
#                 "AvgPointsPerGame": y.tolist(),
#                 "Name": player_names.tolist(),
#                 "RegressionLine": y_pred.tolist()
#             }

#             with open(JSON_DATA_PATH, 'w') as json_file:
#                 json.dump(data, json_file)
#             print(f"Data saved to {JSON_DATA_PATH}")

#             # Add regression predictions to player data
#             player_df['RegressionPoints'] = y_pred

#             return player_df

#         except Exception as e:
#             print(f"Error creating visualization: {e}")
#     else:
#         print("Required columns not found in player data.")

# # Function to normalize player names for matching
# def normalize_name(name):
#     # Remove accents and special characters
#     name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
#     # Remove non-alphanumeric characters except spaces
#     name = re.sub(r'[^a-zA-Z\s]', '', name)
#     # Replace multiple spaces with a single space
#     name = re.sub(r'\s+', ' ', name)
#     # Convert to lowercase
#     return name.strip().lower()

# # Function to find optimal lineups using linear programming
# def find_optimal_lineups(player_df, exclude_players):
#     # Make a copy to avoid modifying the original DataFrame
#     player_df = player_df.copy()

#     # Normalize player names for matching
#     player_df['Normalized_Name'] = player_df['Name'].apply(normalize_name)

#     # Normalize and process exclude_players input
#     exclude_players = [normalize_name(name) for name in exclude_players.split(",") if name.strip()]
#     print(f"Excluding players: {exclude_players}")

#     # Check for unmatched player names
#     unmatched_players = [name for name in exclude_players if name not in player_df['Normalized_Name'].values]
#     if unmatched_players:
#         print(f"The following players were not found and could not be excluded: {unmatched_players}")

#     # Ensure columns are numeric
#     player_df['Salary'] = pd.to_numeric(player_df['Salary'].replace(',', '', regex=True), errors='coerce')
#     player_df['AvgPointsPerGame'] = pd.to_numeric(player_df['AvgPointsPerGame'], errors='coerce')
#     player_df = player_df.dropna(subset=['Salary', 'AvgPointsPerGame'])

#     # Exclude specified players
#     player_df = player_df[~player_df['Normalized_Name'].isin(exclude_players)]

#     # Create a unique ID for each player
#     player_df['ID'] = player_df.index

#     # Display counts of available players by position
#     position_counts = player_df['Position'].value_counts()
#     print("Available players by position:")
#     print(position_counts)

#     # Initialize list to store lineups
#     lineups = []

#     # Generate multiple optimal lineups
#     for lineup_number in range(3):
#         # Define the optimization problem
#         prob = pulp.LpProblem(f"DraftKings_Lineup_Optimization_{lineup_number+1}", pulp.LpMaximize)

#         # Create binary variables for each player
#         player_vars = pulp.LpVariable.dicts("Player", player_df['ID'], cat='Binary')

#         # Objective function: Maximize total average points per game
#         prob += pulp.lpSum([player_vars[player_id] * player_df.loc[player_id, 'AvgPointsPerGame'] for player_id in player_df['ID']])

#         # Salary cap constraint
#         prob += pulp.lpSum([player_vars[player_id] * player_df.loc[player_id, 'Salary'] for player_id in player_df['ID']]) <= 50000

#         # Total players constraint
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df['ID']]) == 9

#         # Position constraints
#         # QB constraint
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'QB']['ID']]) == 1

#         # RB constraints (2 to 3)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'RB']['ID']]) >= 2
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'RB']['ID']]) <= 3

#         # WR constraints (3 to 4)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'WR']['ID']]) >= 3
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'WR']['ID']]) <= 4

#         # TE constraints (1 to 2)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'TE']['ID']]) >= 1
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'TE']['ID']]) <= 2

#         # FLEX position constraint (total RB/WR/TE should be 7)
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'].isin(['RB', 'WR', 'TE'])]['ID']]) == 7

#         # DST constraint
#         prob += pulp.lpSum([player_vars[player_id] for player_id in player_df[player_df['Position'] == 'DST']['ID']]) == 1

#         # Ensure diversity among generated lineups
#         if lineups:
#             for prev_lineup in lineups:
#                 prob += pulp.lpSum([player_vars[player['ID']] for player in prev_lineup]) <= 8  # At least one player different

#         # Solve the optimization problem
#         prob.solve()

#         # Check if an optimal solution was found
#         if pulp.LpStatus[prob.status] != 'Optimal':
#             print(f"No optimal solution found for lineup {lineup_number+1}.")
#             continue

#         # Extract the lineup from the solution
#         lineup = []
#         for player_id in player_df['ID']:
#             if player_vars[player_id].varValue == 1:
#                 player_info = player_df.loc[player_id].to_dict()
#                 lineup.append(player_info)

#         # Verify salary cap
#         total_salary = sum(player['Salary'] for player in lineup)
#         if total_salary > 50000:
#             print(f"Lineup {lineup_number+1} exceeds the salary cap.")
#             continue

#         # Log the lineup details
#         lineup_names = [player['Name'] for player in lineup]
#         print(f"Lineup {lineup_number+1} players: {lineup_names}")
#         print(f"Total salary for lineup {lineup_number+1}: {total_salary}")
#         print(f"Lineup {lineup_number+1} added.")
#         lineups.append(lineup)

#     print(f"Total lineups generated: {len(lineups)}")
#     return lineups  # Return the list of generated lineups

# # Flask route to get player details as JSON
# @app.route('/api/players', methods=['GET'])
# def get_players():
#     try:
#         player_df = parse_player_csv()
#         if player_df is not None:
#             return jsonify(player_df.to_dict(orient='records'))
#         return jsonify({"error": "Player data not available."}), 404
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Flask route to render the visualization page
# @app.route('/visualization', methods=['GET'])
# def render_visualization():
#     try:
#         return render_template('visualization.html', json_data_path=url_for('static', filename='points_vs_salary_data.json'))
#     except Exception as e:
#         return f"An error occurred: {e}", 500

# # Flask route to get optimal lineups as JSON
# @app.route('/api/optimal_lineups', methods=['GET'])
# def get_optimal_lineups():
#     try:
#         player_df = parse_player_csv()
#         exclude_players = request.args.get('exclude_players', '')
#         print(f"Received exclude_players: {exclude_players}")
#         if player_df is not None and not player_df.empty:
#             lineups = find_optimal_lineups(player_df, exclude_players)
#             return jsonify(lineups)
#         return jsonify({"error": "Player data not available or insufficient data."}), 404
#     except Exception as e:
#         print(f"Exception in get_optimal_lineups: {e}")
#         return jsonify({"error": str(e)}), 500

# # Flask route to process the contest link submitted by the user
# @app.route('/api/process_contest_link', methods=['POST'])
# def process_contest_link():
#     try:
#         contest_url = request.json.get('contest_url')
#         if contest_url:
#             contest_type_id, draft_group_id = extract_contest_ids(contest_url)
#             if contest_type_id and draft_group_id:
#                 download_player_csv(contest_type_id, draft_group_id)
#                 # After downloading, create visualizations to update the plot
#                 player_df = parse_player_csv()
#                 if player_df is not None and not player_df.empty:
#                     create_visualizations(player_df)
#                 return jsonify({"message": "Contest data successfully fetched."}), 200
#             else:
#                 return jsonify({"error": "Invalid contest URL format."}), 400
#         else:
#             return jsonify({"error": "Contest URL not provided."}), 400
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     # Run the Flask app in debug mode for development
#     app.run(debug=True, host='0.0.0.0', port=5001)