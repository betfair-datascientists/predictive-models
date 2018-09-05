'''
This script will walk you through creating features from our dataset, which was cleaned in the first tutorial.

If you would like to skip straight to modelling, all you will need to do is import prepare_afl_features from this notebook.
'''

# Import modules
from afl_data_cleaning_v2 import *
import afl_data_cleaning_v2
import pandas as pd
pd.set_option('display.max_columns', None)
from afl_data_cleaning import prepare_match_results
import warnings
warnings.filterwarnings('ignore')
import numpy as np


# Define a function which returns a DataFrame with the expontential moving average for each numeric stat
def create_exp_weighted_avgs(df, span):
    # Create a copy of the df with only the game id and the team - we will add cols to this df
    ema_features = df[['game', 'team']].copy()
    
    feature_names = [col for col in df.columns if col.startswith('f_')] # Get a list of columns we will iterate over
    
    for feature_name in feature_names:
        feature_ema = (df.groupby('team')[feature_name]
                         .transform(lambda row: (row.ewm(span=span)
                                                    .mean()
                                                    .shift(1))))
        ema_features[feature_name] = feature_ema
    
    return ema_features


def create_efficiency_features(afl_data):
	# Get each match on single rows
	single_row_df = (afl_data[['game', 'team', 'f_I50', 'f_R50', 'f_D', 'f_ED', 'home_game', ]]
	                    .query('home_game == 1')
	                    .rename(columns={'team': 'home_team', 'f_I50': 'f_I50_home', 'f_R50': 'f_R50_home', 'f_D': 'f_D_home', 'f_ED': 'f_ED_home'})
	                    .drop(columns='home_game')
	                    .pipe(pd.merge, afl_data[['game', 'team', 'f_I50', 'f_R50', 'f_D', 'f_ED', 'home_game']]
	                                    .query('home_game == 0')
	                                    .rename(columns={'team': 'away_team', 'f_I50': 'f_I50_away', 'f_R50': 'f_R50_away', 'f_D': 'f_D_away', 'f_ED': 'f_ED_away'})
	                                    .drop(columns='home_game'), on='game'))

	single_row_df = single_row_df.assign(f_I50_efficiency_home=lambda df: df.f_R50_away / df.f_I50_home,
                                    f_I50_efficiency_away=lambda df: df.f_R50_home / df.f_I50_away)

	feature_efficiency_cols = ['f_I50_efficiency_home', 'f_I50_efficiency_away']

	# Now let's create an Expontentially Weighted Moving Average for these features - we will need to reshape our DataFrame to do this
	efficiency_features_multi_row = (single_row_df[['game', 'home_team'] + feature_efficiency_cols]
	                                    .rename(columns={
	                                        'home_team': 'team',
	                                        'f_I50_efficiency_home': 'f_I50_efficiency',
	                                        'f_I50_efficiency_away': 'f_I50_efficiency_opponent',
	                                    })
	                                    .append((single_row_df[['game', 'away_team'] + feature_efficiency_cols]
	                                                 .rename(columns={
	                                                     'away_team': 'team',
	                                                     'f_I50_efficiency_home': 'f_I50_efficiency_opponent',
	                                                     'f_I50_efficiency_away': 'f_I50_efficiency',
	                                                 })), sort=True)
	                                    .sort_values(by='game')
	                                    .reset_index(drop=True))

	efficiency_features = efficiency_features_multi_row[['game', 'team']].copy()
	feature_efficiency_cols = ['f_I50_efficiency', 'f_I50_efficiency_opponent']

	for feature in feature_efficiency_cols:
	    efficiency_features[feature] = (efficiency_features_multi_row.groupby('team')[feature]
	                                        .transform(lambda row: row.ewm(span=10).mean().shift(1)))
	    
	# Get feature efficiency df back onto single rows
	efficiency_features = pd.merge(efficiency_features, afl_data[['game', 'team', 'home_game']], on=['game', 'team'])
	efficiency_features_single_row = (efficiency_features.query('home_game == 1')
	                                    .rename(columns={
	                                        'team': 'home_team', 
	                                        'f_I50_efficiency': 'f_I50_efficiency_home',
	                                        'f_I50_efficiency_opponent': 'f_R50_efficiency_home'})
	                                    .drop(columns='home_game')
	                                    .pipe(pd.merge, (efficiency_features.query('home_game == 0')
	                                                        .rename(columns={
	                                                            'team': 'away_team',
	                                                            'f_I50_efficiency': 'f_I50_efficiency_away',
	                                                            'f_I50_efficiency_opponent': 'f_R50_efficiency_away'})
	                                                        .drop(columns='home_game')), on='game'))
	return efficiency_features_single_row


# Define a function which finds the elo for each team in each game and returns a dictionary with the game ID as a key and the
# elos as the key's value, in a list. It also outputs the probabilities and a dictionary of the final elos for each team
def elo_applier(df, k_factor):
    # Initialise a dictionary with default elos for each team
    elo_dict = {team: 1500 for team in df['team'].unique()}
    elos, elo_probs = {}, {}
    
    # Get a home and away dataframe so that we can get the teams on the same row
    home_df = df.loc[df.home_game == 1, ['team', 'game', 'f_margin', 'home_game']].rename(columns={'team': 'home_team'})
    away_df = df.loc[df.home_game == 0, ['team', 'game']].rename(columns={'team': 'away_team'})
    
    df = (pd.merge(home_df, away_df, on='game')
            .sort_values(by='game')
            .drop_duplicates(subset='game', keep='first')
            .reset_index(drop=True))

    # Loop over the rows in the DataFrame
    for index, row in df.iterrows():
        # Get the Game ID
        game_id = row['game']
        
        # Get the margin
        margin = row['f_margin']
        
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
        if margin > 0: # Home team wins; update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away)
        elif margin < 0: # Away team wins; update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away)
        elif margin == 0: # Drawn game' update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(0.5 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0.5 - prob_win_away)
        
        # Update elos in elo dictionary
        elo_dict[home_team] = new_home_team_elo
        elo_dict[away_team] = new_away_team_elo
    
    return elos, elo_probs, elo_dict


def create_form_features(match_results=None):
	if match_results is None:
		match_results = afl_data_cleaning_v2.get_cleaned_match_results()

	form_btwn_teams = match_results[['game', 'team', 'opponent', 'margin']].copy()

	form_btwn_teams['f_form_margin_btwn_teams'] = (match_results.groupby(['team', 'opponent'])['margin']
	                                                          .transform(lambda row: row.rolling(5).mean().shift())
	                                                          .fillna(0))

	form_btwn_teams['f_form_past_5_btwn_teams'] = \
	(match_results.assign(win=lambda df: df.apply(lambda row: 1 if row.margin > 0 else 0, axis='columns'))
	              .groupby(['team', 'opponent'])['win']
	              .transform(lambda row: row.rolling(5).mean().shift() * 5)
	              .fillna(0))
	return form_btwn_teams



def prepare_afl_features(afl_data=None, match_results=None, odds=None):
	# Use the prepare_afl_data function to prepare the data for us; this function condenses what we walked through in the previous tutorial
	if afl_data is None:
		afl_data = prepare_afl_data()

	features = afl_data[['date', 'game', 'team', 'opponent', 'venue', 'home_game']].copy()
	features_rolling_averages = create_exp_weighted_avgs(afl_data, span=10)
	features = pd.merge(features, features_rolling_averages, on=['game', 'team'])
	
	# If we are predicting future rounds we need to append future round data to match_results to create features
	if match_results is None:
		form_btwn_teams = create_form_features()
	else:
		form_btwn_teams = create_form_features(match_results)
	
	# Merge to our features df
	features = pd.merge(features, form_btwn_teams.drop(columns=['margin']), on=['game', 'team', 'opponent'])

	# Use the elo applier function to get the elos and elo probabilities for each game - we will map these later
	elos, probs, elo_dict = elo_applier(afl_data, 30)

	one_line_cols = ['game', 'team', 'home_game'] + [col for col in features if col.startswith('f_')]

	# Get all features onto individual rows for each match
	features_one_line = (features.loc[features.home_game == 1, one_line_cols]
	                     .rename(columns={'team': 'home_team'})
	                     .drop(columns='home_game')
	                     .pipe(pd.merge, (features.loc[features.home_game == 0, one_line_cols]
	                                              .drop(columns='home_game')
	                                              .rename(columns={'team': 'away_team'})
	                                              .rename(columns={col: col+'_away' for col in features.columns if col.startswith('f_')})), on='game')
	                     .drop(columns=['f_form_margin_btwn_teams_away', 'f_form_past_5_btwn_teams_away']))

	# Create efficiency features
	efficiency_features_single_row = create_efficiency_features(afl_data)

	# Add our created features - elo, efficiency etc.
	features_one_line = (features_one_line.assign(f_elo_home=lambda df: df.game.map(elos).apply(lambda x: x[0]),
	                                            f_elo_away=lambda df: df.game.map(elos).apply(lambda x: x[1]))
	                                      .pipe(pd.merge, efficiency_features_single_row, on=['game', 'home_team', 'away_team'])
	                                      .pipe(pd.merge, afl_data.loc[afl_data.home_game == 1, ['game', 'date', 'round', 'venue']], on=['game'])
	                                      .dropna()
	                                      .reset_index(drop=True)
	                                      .assign(season=lambda df: df.date.apply(lambda row: row.year)))


	# Order the columns so that the game info is on the left
	ordered_cols = [col for col in features_one_line if col[:2] != 'f_'] + [col for col in features_one_line if col.startswith('f_')]
	feature_df = features_one_line[ordered_cols]


	# Create differential df - this df is the home features - the away features
	diff_cols = [col for col in feature_df.columns if col + '_away' in feature_df.columns and col != 'f_odds' and col.startswith('f_')]
	non_diff_cols = [col for col in feature_df.columns if col not in diff_cols and col[:-5] not in diff_cols]

	diff_df = feature_df[non_diff_cols].copy()

	for col in diff_cols:
	    diff_df[col+'_diff'] = feature_df[col] - feature_df[col+'_away']

	# Add current odds in to diff_df
	if odds is None:
		odds = get_cleaned_odds()

	home_odds = (odds[odds.home_game == 1]
	             .assign(f_current_odds_prob=lambda df: 1 / df.odds)
	             .rename(columns={'team': 'home_team'})
	             .drop(columns=['home_game', 'odds']))

	away_odds = (odds[odds.home_game == 0]
	             .assign(f_current_odds_prob_away=lambda df: 1 / df.odds)
	             .rename(columns={'team': 'away_team'})
	             .drop(columns=['home_game', 'odds']))

	diff_df = (diff_df.pipe(pd.merge, home_odds, on=['date', 'home_team'])
	              .pipe(pd.merge, away_odds, on=['date', 'away_team']))

	return diff_df