# Import Modules
from afl_feature_creation_v2 import prepare_afl_features
import afl_data_cleaning_v2
import afl_feature_creation_v2
import afl_modelling
import datetime
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Define a function which grabs the odds for each game for the following weekend
def get_next_week_odds(path):
    # Get next week's odds
    next_week_odds = pd.read_csv(path)
    next_week_odds = next_week_odds.rename(columns={"team_1": "home_team", 
                                                "team_2": "away_team", 
                                                "team_1_odds": "odds", 
                                                "team_2_odds": "odds_away"
                                               })
    return next_week_odds


# Import the fixture
# Define a function which gets the fixture and cleans it up
def get_fixture(path):
    # Get the afl fixture
    fixture = pd.read_csv(path)

    # Replace team names and reformat
    fixture = fixture.replace({'Brisbane Lions': 'Brisbane', 'Footscray': 'Western Bulldogs'})
    fixture['Date'] = pd.to_datetime(fixture['Date']).dt.date.astype(str)
    fixture = fixture.rename(columns={"Home.Team": "home_team", "Away.Team": "away_team"})
    return fixture


# Define a function which grabs the odds for each game for the following weekend
def get_next_week_odds(path):
    # Get next week's odds
    next_week_odds = pd.read_csv(path)
    next_week_odds = next_week_odds.rename(columns={"team_1": "home_team", 
                                                "team_2": "away_team", 
                                                "team_1_odds": "odds", 
                                                "team_2_odds": "odds_away"
                                               })
    return next_week_odds


# Import the fixture
# Define a function which gets the fixture and cleans it up
def get_fixture(path):
    # Get the afl fixture
    fixture = pd.read_csv(path)

    # Replace team names and reformat
    fixture = fixture.replace({'Brisbane Lions': 'Brisbane', 'Footscray': 'Western Bulldogs'})
    fixture['Date'] = pd.to_datetime(fixture['Date']).dt.date.astype(str)
    fixture = fixture.rename(columns={"Home.Team": "home_team", "Away.Team": "away_team"})
    return fixture


# Define a function which creates game IDs for this week's footy games
def create_next_weeks_game_ids(afl_data):
    odds = get_next_week_odds("data/weekly_odds.csv")

    # Get last week's Game ID
    last_afl_data_game = afl_data['game'].iloc[-1]

    # Create Game IDs for next week
    game_ids = [(i+1) + last_afl_data_game for i in range(odds.shape[0])]
    return game_ids


# Define a function which creates this week's footy game DataFrame
def get_next_week_df(afl_data):
    # Get the fixture and the odds for next week's footy games
    fixture = get_fixture("data/afl_fixture_2018.csv")
    next_week_odds = get_next_week_odds("data/weekly_odds.csv")
    next_week_odds['game'] = create_next_weeks_game_ids(afl_data)

    # Get today's date and next week's date and create a DataFrame for next week's games
    todays_date = datetime.datetime.today().strftime('%Y-%m-%d')
    date_in_7_days = (datetime.datetime.today() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
    fixture = fixture[(fixture['Date'] >= todays_date) & (fixture['Date'] < date_in_7_days)].drop(columns=['Season.Game'])
    next_week_df = pd.merge(fixture, next_week_odds, on=['home_team', 'away_team'])

    # Split the DataFrame onto two rows for each game
    h_df = (next_week_df[['Date', 'game', 'home_team', 'away_team', 'odds', 'Season', 'Round', 'Venue']]
               .rename(columns={'home_team': 'team', 'away_team': 'opponent'})
               .assign(home_game=1))

    a_df = (next_week_df[['Date', 'game', 'home_team', 'away_team', 'odds_away', 'Season', 'Round', 'Venue']]
                .rename(columns={'odds_away': 'odds', 'home_team': 'opponent', 'away_team': 'team'})
                .assign(home_game=0))

    next_week = a_df.append(h_df).sort_values(by='game').rename(columns={
        'Date': 'date',
        'Season': 'season',
        'Round': 'round',
        'Venue': 'venue'
    })
    next_week['date'] = pd.to_datetime(next_week.date)
    next_week['round'] = afl_data['round'].iloc[-1] + 1
    return next_week


def create_predictions():
	# Grab the cleaned AFL dataset and the column order
	afl_data = afl_data_cleaning_v2.prepare_afl_data()
	ordered_cols = afl_data.columns

	next_week_odds = get_next_week_odds("data/weekly_odds.csv")
	fixture = get_fixture("data/afl_fixture_2018.csv")

	next_week_df = get_next_week_df(afl_data)
	game_ids_next_round = create_next_weeks_game_ids(afl_data)
	next_week_df

	# Append next week's games to our afl_data DataFrame
	afl_data = afl_data.append(next_week_df).reset_index(drop=True)

	# Append next week's games to match results (we need to do this for our feature creation to run)
	match_results = afl_data_cleaning_v2.get_cleaned_match_results().append(next_week_df)

	# Append next week's games to odds
	odds = (afl_data_cleaning_v2.get_cleaned_odds().pipe(lambda df: df.append(next_week_df[df.columns]))
	       .reset_index(drop=True))

	features_df = afl_feature_creation_v2.prepare_afl_features(afl_data=afl_data, match_results=match_results, odds=odds)

	# Get the train df by only taking the games IDs which aren't in the next week df
	train_df = features_df[~features_df.game.isin(next_week_df.game)]

	# Get the result and merge to the feature_df
	match_results = (pd.read_csv("data/afl_match_results.csv")
	                    .rename(columns={'Game': 'game'})
	                    .assign(result=lambda df: df.apply(lambda row: 1 if row['Home.Points'] > row['Away.Points'] else 0, axis=1)))

	train_df = pd.merge(train_df,  match_results[['game', 'result']], on='game')

	train_x = train_df.drop(columns=['result'])
	train_y = train_df.result

	next_round_x = features_df[features_df.game.isin(next_week_df.game)]

	# Fit out logistic regression model - note that our predictions come out in the order of [away_team_prob, home_team_prob]

	lr_best_params = {'C': 0.01,
	 'class_weight': None,
	 'dual': False,
	 'fit_intercept': True,
	 'intercept_scaling': 1,
	 'max_iter': 100,
	 'multi_class': 'ovr',
	 'n_jobs': 1,
	 'penalty': 'l2',
	 'random_state': None,
	 'solver': 'newton-cg',
	 'tol': 0.0001,
	 'verbose': 0,
	 'warm_start': False}

	feature_cols = [col for col in train_df if col.startswith('f_')]

	# Scale features
	scaler = StandardScaler()
	train_x[feature_cols] = scaler.fit_transform(train_x[feature_cols])
	next_round_x[feature_cols] = scaler.transform(next_round_x[feature_cols])

	lr = LogisticRegression(**lr_best_params)
	lr.fit(train_x[feature_cols], train_y)
	prediction_probs = lr.predict_proba(next_round_x[feature_cols])

	modelled_home_odds = [1/i[1] for i in prediction_probs]
	modelled_away_odds = [1/i[0] for i in prediction_probs]

	# Create a predictions df
	preds_df = (next_round_x[['date', 'home_team', 'away_team', 'venue', 'game']].copy()
	               .assign(modelled_home_odds=modelled_home_odds,
	                      modelled_away_odds=modelled_away_odds)
	               .pipe(pd.merge, next_week_odds, on=['home_team', 'away_team'])
	               .pipe(pd.merge, features_df[['game', 'f_elo_home', 'f_elo_away']], on='game')
	               .drop(columns='game')
	           )
	return preds_df