'''
This script will walk you through creating features from our dataset, which was cleaned in the first tutorial.

If you would like to skip straight to modelling, all you will need to do is import prepare_afl_features from this notebook.
'''

# Import libraries
from afl_data_cleaning import prepare_afl_data
from afl_data_cleaning import prepare_match_results
import pandas as pd
import numpy as np
import afl_data_cleaning
import warnings
warnings.filterwarnings('ignore')

# First we will define a function which creates a new DataFrame with Opposition Statistics on the same row as the Team Statistics 
def get_opp_stats_df(df):
    # Filter the DataFrames by whether it was a home or away game
    home_df = df[df['Status'] == 'Home']
    away_df = df[df['Status'] == 'Away']

    # Rename the away columns so we know that they are from the away team
    home_df_renamed = home_df.rename(columns={col: col + '_Opp' for col in home_df.columns if col != 'Game'})
    away_df_renamed = away_df.rename(columns={col: col + '_Opp' for col in home_df.columns if col != 'Game'})

    # Merge the two DataFrames on the Game
    merged_1 = pd.merge(home_df, away_df_renamed, on=['Game'])
    merged_2 = pd.merge(away_df, home_df_renamed, on=['Game'])

    # Append the DataFrames together and then sort by the Game Id, reset the index and drop unrequired columns
    merged = merged_1.append(merged_2).sort_values(by='Game').reset_index(drop=True)
    return merged

# Creating Efficiency Features
def create_efficiency_features(df):
	# Create Disposal Efficiency feature - the proportion of disposals which are 'effective disposals'
	df['disposal_efficiency'] = df['ED'] / df['D']

	# Create Rebound 50 Efficiency feature - we will first need to grab the opposition's stats using our get_opp_stats_df function
	opponent_stats_df = get_opp_stats_df(df)

	# Create Rebound 50 Efficiency - the proportion of Rebound 50s from opposition Inside 50s
	opponent_stats_df['R50_efficiency'] = opponent_stats_df['R50'] / opponent_stats_df['I50_Opp']

	# Create Inside 50 Efficiency - the proportion of opposition Rebound 50s from Inside 50s
	opponent_stats_df['I50_efficiency'] = opponent_stats_df['R50_Opp'] / opponent_stats_df['I50']

	# Merge features to main df DataFrame
	df = pd.merge(df, opponent_stats_df[['Team', 'Game', 'R50_efficiency', 'I50_efficiency']], on=['Team', 'Game'])
	return df

# Define a function which returns a DataFrame with the rolling averages for each game. Cols refers to the columns which we want
# to create a rolling average for
def create_rolling_averages(df, window, cols):
	new_cols = [col + '_ave_{}'.format(window) for col in cols]
	df[new_cols] = df.groupby('Team')[cols].apply(lambda x: x.rolling(window).mean().shift())
	df = df.drop(columns=cols)
	return df

# Define a function which finds the elo for each team in each game and returns a dictionary with the game ID as a key and the
# elos as the key's value, in a list. It also outputs the probabilities and a dictionary of the final elos for each team
def elo_applier(df, k_factor):
    # Initialise a dictionary with default elos for each team
    elo_dict = {team: 1500 for team in df['Team'].unique()}
    elos, elo_probs = {}, {}
    
    # Sort by Game and then only grab the Home Games so that the same isn't repeated
    df = df.sort_values(by=['Game']).reset_index(drop=True)
    df = df[df['Home?'] == 1]
    df = df.drop_duplicates(subset='Game', keep='first')
    
    # Loop over the rows in the DataFrame
    for index, row in df.iterrows():
        # Get the Game ID
        game_id = row['Game']
        
        # If the game already has the elos for the home and away team in the elos dictionary, go to the next game
        if game_id in elos.keys():
            continue
        
        # Get the team and opposition
        home_team = row['home_team']
        away_team = row['away_team']
        
        # Get the team and opposition elo score
        home_team_elo = elo_dict[home_team]
        away_team_elo = elo_dict[away_team]
        
        # Calculated the probability of winning for the team and opposition
        prob_win_home = 1 / (1 + 10**((away_team_elo - home_team_elo) / 400))
        prob_win_away = 1 - prob_win_home
        
        # Add the elos and probabilities our elos dictionary and elo_probs dictionary based on the Game ID
        elos[game_id] = [home_team_elo, away_team_elo]
        elo_probs[game_id] = [prob_win_home, prob_win_away]
        
        # Calculate the new elos of each team
        if row['Margin'] > 0: # Team wins; update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away)
        elif row['Margin'] < 0: # Away team wins; update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away)
        elif row['Margin'] == 0: # Drawn game' update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(0.5 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0.5 - prob_win_away)
        
        # Update elos in elo dictionary
        elo_dict[home_team] = new_home_team_elo
        elo_dict[away_team] = new_away_team_elo
    
    return elos, elo_probs, elo_dict

# Create a function which maps the current home and away team elos to 'Team Elo' and 'Opposition Elo'
def map_elos(df):
    home_df = df[df['Home?'] == 1]
    away_df = df[df['Home?'] == 0]
    home_df['elo'] = home_df['home_elo']
    home_df['elo_Opp'] = home_df['away_elo']
    away_df['elo'] = away_df['away_elo']
    away_df['elo_Opp'] = away_df['home_elo']
    final_df = home_df.append(away_df).sort_values(by=['Game', 'Home?'])
    return final_df

# Define a function which calculates the form between teams over a given window
def form_between_teams(df, window):
    num_wins_over_opposition = []
    # Iterate over rows
    for idx, row in df.iterrows():
        # Get a DataFrame of recent games between the teams
        recent_games_between_teams = df[(df['Team'] == row['Team']) & (df['Opposition'] == row['Opposition'])].loc[:idx-1][-window:]
        # Calculate the number of wins for the current team
        num_wins = recent_games_between_teams[recent_games_between_teams['Margin'] > 0].shape[0]
        # Append this to a list
        num_wins_over_opposition.append(num_wins)
    # Add the new feature as a column
    df['form_over_opposition_{}'.format(window)] = num_wins_over_opposition
    return df

def prepare_form_between_teams(df, window=5):
	# Grab the match_results DataFrame to look at historical form
	match_results = prepare_match_results("data/afl_match_results.csv")

	# Filter for games after 2004
	match_results = match_results.iloc[25000:].reset_index(drop=True)

	# Find the historical form between teams and create a new DataFrame with the form_over_opposition feature in it
	form = form_between_teams(match_results, window)[['Date', 'Team', 'Opposition', 'form_over_opposition_{}'.format(window)]]

	# Join the new DataFrame to our main DataFrame
	df = pd.merge(df, form, on=['Date', 'Team', 'Opposition'])
	return df

# Create a feature which calculates the average elo of opponents when a team has won and lost
def create_ave_elo_opponent(df, window, beaten_or_lost='beaten'):
    elos_of_opponents = {team: [] for team in df['Team'].unique()}
    ave_elo_opponents = []
    
    # Loop over rows of the DataFrame
    for idx, row in df.iterrows():
        # Grab the mean elos of opponents beaten and append it to a list
        if len(elos_of_opponents[row['Team']]) >= window:
            ave_elo_opponents.append(np.mean(elos_of_opponents[row['Team']][-window:]))
        else:
            ave_elo_opponents.append(np.nan)
        
        if beaten_or_lost == 'beaten':
            # Update the elos of opponents beaten for this game (if the team wins, add their opponents elo to the dictionary)
            if row['Margin'] > 0 and row['Home?'] == 1:
                elos_of_opponents[row['Team']].append(row['away_elo']) 
            elif row['Margin'] > 0 and row['Home?'] == 0:
                elos_of_opponents[row['Team']].append(row['home_elo'])
        
        elif beaten_or_lost == 'lost':
            # Update the elos of opponents lost to for this game (if the team wins, add their opponents elo to the dictionary)
            if row['Margin'] < 0 and row['Home?'] == 1:
                elos_of_opponents[row['Team']].append(row['away_elo']) 
            elif row['Margin'] < 0 and row['Home?'] == 0:
                elos_of_opponents[row['Team']].append(row['home_elo'])
                
    df['average_elo_opponents_{}_{}'.format(beaten_or_lost, window)] = ave_elo_opponents
    return df

# Define our final function, which uses all the other functions defined to create our feature set
def prepare_afl_features(window=6, k_factor=24):
	# Grab the cleaned dataset which we prepared in the first tutorial 
	afl_data = prepare_afl_data().drop_duplicates()

	# Create efficiency features, such as Disposal Efficiency, Inside 50 Efficiency etc.
	afl_data = create_efficiency_features(afl_data)

	# Create rolling average features
	cols_indx_start = afl_data.columns.get_loc("GA")
	afl_avgs = create_rolling_averages(afl_data, window, afl_data.columns[cols_indx_start:])

	# Create elo features - map the elos to our DataFrame
	elos, probs, elo_dict = elo_applier(afl_avgs, k_factor)
	afl_avgs['home_elo'] = afl_avgs['Game'].map(elos).apply(lambda x: x[0])
	afl_avgs['away_elo'] = afl_avgs['Game'].map(elos).apply(lambda x: x[1])

	# Create elo adjusted average margin feature
	# Use our map_elos function to get the elo for each team on each row and their opposition
	afl_avgs = map_elos(afl_avgs)

	# Create Adjusted Margin and then Average it over our window. Also create a regular rolling average for Margin
	afl_avgs['Adj_elo_ave_margin'] = afl_avgs['Margin'] * afl_avgs['elo_Opp'] / afl_avgs['elo']
	afl_avgs = create_rolling_averages(afl_avgs, window, ['Adj_elo_ave_margin'])

	# Create a form between the teams feature
	afl_avgs = prepare_form_between_teams(afl_avgs, window=6)

	# Create an average elo of opponent when having won and lost, over a given window
	afl_avgs = create_ave_elo_opponent(afl_avgs, window, beaten_or_lost='beaten')
	afl_avgs = create_ave_elo_opponent(afl_avgs, window, beaten_or_lost='lost')

	# Create the home_win column - this will be our target variable
	afl_avgs['home_win'] = afl_avgs.apply(lambda x: 1 if x['Margin'] > 0 else 0, axis=1)

	# Finally, create a rolling average of the margin
	afl_avgs = create_rolling_averages(afl_avgs, window, ['Margin'])

	return afl_avgs