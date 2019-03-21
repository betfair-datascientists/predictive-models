# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss
from trueskill import Rating
import trueskill
import math
from trueskill import BETA
from trueskill.backends import cdf
import h2o
from h2o.automl import H2OAutoML
import betfairlightweight
import os
from betfairlightweight import APIClient
from betfairlightweight import filters
import pytz
import datetime
import boto3
import json


def multiplayer_trueskill_applier(df, race_id_col, player1_col, player2_col, starting_mu=25.0, starting_std=25.0/3.0):
    env = trueskill.TrueSkill(draw_probability=0.015)
    # Get a list of unique players
    unique_players = df[player1_col].unique().tolist() + df[player2_col].unique().tolist()

    # Initialise a dictionary with default elos for each player
    ratings_dict = {team: Rating(mu=starting_mu, sigma=starting_std) for team in unique_players}
    
    # Create dict where we will update based on the race_id and horse_id
    before_game_ratings = []
    after_game_ratings = []
    # Loop over races in each group
    for idx, row in df.iterrows():
       
        before_game_ratings.append([ratings_dict[row[player1_col]], ratings_dict[row[player2_col]]])
        
        if row.winner_1 == 1:
            new_r1, new_r2 = trueskill.rate_1vs1(ratings_dict[row[player1_col]], ratings_dict[row[player2_col]])
            ratings_dict[row[player1_col]] = new_r1
            ratings_dict[row[player2_col]] = new_r2
        elif row.winner_2 == 1:
            new_r1, new_r2 = trueskill.rate_1vs1(ratings_dict[row[player2_col]], ratings_dict[row[player1_col]])
            ratings_dict[row[player1_col]] = new_r2
            ratings_dict[row[player2_col]] = new_r1
        elif row.draw == 1:
            new_r1, new_r2 = trueskill.rate_1vs1(ratings_dict[row[player1_col]], ratings_dict[row[player2_col]], drawn=True)
            ratings_dict[row[player1_col]] = new_r1
            ratings_dict[row[player2_col]] = new_r2
        else:
            print('error')
        after_game_ratings.append([ratings_dict[row[player1_col]], ratings_dict[row[player2_col]]])
    return before_game_ratings, after_game_ratings, ratings_dict


def elo_applier(df, player1_col, player2_col, game_id_col, margin_col, k_factor=25, hga_factor=35):
    # Initialise a dictionary with default elos for each team
    players = df[player2_col].tolist() + df[player1_col].tolist()
    
    elos_dict = {team: 1500 for team in players}

    elos, elo_probs = {}, {}
    
    # Loop over the rows in the DataFrame
    for index, row in df.iterrows():
        
        # Get the team and opposition
        team_1 = row[player1_col]
        team_2 = row[player2_col]
            
        team_1_elo = elos_dict[team_1]
        team_2_elo = elos_dict[team_2]
        
        
        if row.home_team_1 == 1:
            team_1_elo_hga = elos_dict[team_1] + hga_factor
            team_2_elo_hga = elos_dict[team_2]
        elif row.home_team_2 == 1:
            team_1_elo_hga = elos_dict[team_1]
            team_2_elo_hga = elos_dict[team_2] + hga_factor
        
        # Calculated the probability of winning for the team and opposition
        prob_win_team_1 = 1 / (1 + 10**((team_2_elo_hga - team_1_elo_hga) / 400))
        prob_win_team_2 = 1 - prob_win_team_1
        
        # Add the elos and probabilities our elos dictionary and elo_probs dictionary based on the Game ID
        elos[row[game_id_col]] = [team_1_elo, team_2_elo]
        elo_probs[row[game_id_col]] = [prob_win_team_1, prob_win_team_2]
        
        margin = row[margin_col]

        if margin == 0: # Draw
            new_team_1_elo = team_1_elo + k_factor*(0.5 - prob_win_team_1)
            new_team_2_elo = team_2_elo + k_factor*(0.5 - prob_win_team_2) 
#             print("Draw, new elos:", new_team_1_elo, new_team_2_elo)
        elif margin > 0: # Team 1 win
            new_team_1_elo = team_1_elo + k_factor*(1 - prob_win_team_1)
            new_team_2_elo = team_2_elo + k_factor*(0 - prob_win_team_2)
#             print("Team 1 wins, new elos:", new_team_1_elo, new_team_2_elo)
        elif margin < 0:
            new_team_1_elo = team_1_elo + k_factor*(0 - prob_win_team_1)
            new_team_2_elo = team_2_elo + k_factor*(1 - prob_win_team_2)
#             print("Team 2 wins, new elos:", new_team_1_elo, new_team_2_elo)
        
        
        # Update elos in elo dictionary
        elos_dict[team_1] = new_team_1_elo
        elos_dict[team_2] = new_team_2_elo
    return elos, elo_probs, elos_dict


def get_data():
    # Get data
    relevant_cols = ['Date', 'Home Team', 'Away Team', 'Home Score', 'Away Score', 'Home Odds', 'Draw Odds', 'Away Odds']


    rename_dict = {
        'Date': 'date',
        'Home Team': 'home_team',
        'Away Team': 'away_team',
        'Home Score': 'home_score',
        'Away Score': 'away_score',
        'Home Odds': 'home_odds',
        'Draw Odds': 'draw_odds',
        'Away Odds': 'away_odds'
    }

    historic = (pd.read_excel('http://www.aussportsbetting.com/historical_data/nrl.xlsx', header=1)[relevant_cols]
               .rename(columns=rename_dict)
               )
    return historic

def create_json_predictions(predictions_df, game_week):
	json_preds = {
		"tournaments": [
			{
				"name": "NRL",
				"title": "NRL Predictions".format(game_week),
				"season": "2019",
				"round": "{}".format(game_week),
                "roundName": str(game_week),
				"bfExchangeEventId": str(competition_id),
				"blurb": "Comment",
				"matches": [
					{
						"name": str(event_name),
						"date": str(date_time),
						"venue": "NA",
						"pred_margin": "Predicted Winner: {}".format(predicted_winner),
						"bfExchangeEventId": str(event_id),
						"markets": [
							{
								"bfExchangeMarketId": str(market_id),
								"picks": [
									{
										"bfExchangeSelectionId": str(g2.groupby("selection_id_1")['selection_id_1'].sum().values[0]),
										"model": str(round(g2.groupby('model_odds_1')['model_odds_1'].sum().values[0], 2)),
                                        "line": "+0"
									},
									{
										"bfExchangeSelectionId": str(g2.groupby('selection_id_2')['selection_id_2'].sum().values[0]),
										"model": str(round(g2.groupby('model_odds_2')['model_odds_2'].sum().values[0], 2)),
                                        "line": "+0"
									}
								]
							} for market_id, g2 in g.groupby("market_id")
						]
					} for (event_name, event_id, date_time, predicted_winner), g in group.groupby(["event_name", "event_id", "localMarketStartTime", "predicted_winner"])
				]
			} for (competition_id), group in predictions_df.groupby("competition_id")
		]
	}
	return json_preds


def save_to_local_machine(path, json_file):
    with open(path, "w") as f:
        json.dump(json_file, f)

def convert_home_away_df_to_randomised_df(df, both_cols):
    '''
    This method converts a dataframe which has the format of statistics ordered by the home and away teams/players
    to a randomised dataframe with a winner flag column
    
    The home columns must start with "home_" and the away cols must start with "away_"
    '''
    
    df = df.sort_values(by='date').reset_index(drop=True)
    df['GAME_ID'] = df.reset_index(drop=True).index
    
    home_cols = [col for col in df.columns if 'home_' in col]
    away_cols = [col for col in df.columns if 'away_' in col]

    home_df = df[['GAME_ID'] + home_cols].copy()

    # Rename columns
    home_df = home_df.rename(columns=lambda col: col.replace('home_', ''))
    home_df['home_team'] = 1

    away_df = df[['GAME_ID'] + away_cols].copy()

    away_df = away_df.rename(columns=lambda col: col.replace('away_', ''))
    away_df['home_team'] = 0

    # Concat these then randomise
    combined_df = pd.concat([home_df, away_df]).reset_index(drop=True).sample(frac=1)

    df1 = combined_df.sort_values(by='GAME_ID').reset_index(drop=True)[::2]
    df2 = combined_df.sort_values(by='GAME_ID').reset_index(drop=True)[1::2]

    # Merge back together
    final = pd.merge(df1, df2, on='GAME_ID', suffixes=['_1', '_2'])

    # Merge back the match info too
    final = pd.merge(df[['GAME_ID'] + both_cols], final, on='GAME_ID')
    return final

def get_train_features():
    historic = get_data()
    random_df = convert_home_away_df_to_randomised_df(historic, both_cols=['date', 'draw_odds'])

    # Add columns and filter out missing vals
    random_df['margin'] = random_df['score_1'] - random_df['score_2']
    random_df['winner_1'] = np.where(random_df.margin > 0, 1, 0)
    random_df['draw'] = np.where(random_df.margin == 0, 1, 0)
    random_df['winner_2'] = np.where(random_df.margin < 0, 1, 0)
    random_df = random_df[~random_df.score_1.isnull()]
    random_df = random_df.sort_values(by='date')

    # These were optimised

    K_FACTOR = 27.5
    HGA = 20
    ROLLING_AVE_WINDOW = 20

    teams_to_replace = {
        'Brisbane Broncos': 'Brisbane',
        'Canberra Raiders': 'Canberra',
        'Canterbury Bulldogs': 'Canterbury',
        'Canterbury-Bankstown Bulldogs': 'Canterbury',
        'Cronulla Sharks': 'Cronulla',
        'Cronulla-Sutherland Sharks': 'Cronulla',
        'Gold Coast Titans': 'Gold Coast',
        'Manly Sea Eagles': 'Manly',
        'Manly-Warringah Sea Eagles': 'Manly',
        'Melbourne Storm': 'Melbourne',
        'New Zealand Warriors': 'NZ Warriors',
        'Newcastle Knights': 'Newcastle',
        'North QLD Cowboys': 'North Qld',
        'North Queensland Cowboys': 'North Qld',
        'Parramatta Eels': 'Parramatta',
        'Penrith Panthers': 'Penrith',
        'South Sydney Rabbitohs': 'South Sydney',
        'St George Dragons': 'St George',
        'St. George Illawarra Dragons': 'St George',
        'Sydney Roosters': 'Sydney',
        'Wests Tigers': 'Wests Tigers'
}
    
    random_df['team_1'] = random_df['team_1'].replace(teams_to_replace)
    random_df['team_2'] = random_df['team_2'].replace(teams_to_replace)
    
    before_game_trueskill_ratings, after_game_trueskill_ratings, trueskill_ratings_dict = \
    multiplayer_trueskill_applier(random_df, 
                                  'GAME_ID',
                                  'team_1',
                                  'team_2')
    
    elos, elo_probs, elos_dict = \
    elo_applier(df=random_df,
                player1_col='team_1',
                player2_col='team_2',
                game_id_col='GAME_ID',
                margin_col='margin',
                k_factor=K_FACTOR, 
                hga_factor=HGA)
    
    random_df['elo_1'] = random_df.GAME_ID.map(elos).str[0]
    random_df['elo_2'] = random_df.GAME_ID.map(elos).str[1]
    random_df['elo_prob_1'] = random_df.GAME_ID.map(elo_probs).str[0]
    random_df['elo_prob_2'] = random_df.GAME_ID.map(elo_probs).str[1]
    random_df['elo_prob_diff'] = random_df.GAME_ID.map(elo_probs).str[0] - random_df.GAME_ID.map(elo_probs).str[1]
    random_df['trueskill_mu_1'] = [rating[0].mu for rating in before_game_trueskill_ratings]
    random_df['trueskill_mu_2'] = [rating[1].mu for rating in before_game_trueskill_ratings]
    random_df['trueskill_sigma_1'] = [rating[0].sigma for rating in before_game_trueskill_ratings]
    random_df['trueskill_sigma_2'] = [rating[1].sigma for rating in before_game_trueskill_ratings]

    # Split df into long
    df1 = random_df[['GAME_ID', 'date', 'team_1', 'margin']].rename(columns={'team_1': 'team'})
    df2 = random_df[['GAME_ID', 'date', 'team_2', 'margin']].rename(columns={'team_2': 'team'}).assign(margin=lambda df: df.margin*-1)

    long_df = pd.concat([df1, df2], sort=False).sort_values(by='GAME_ID').reset_index(drop=True)
    long_df['ave_margin'] = long_df.groupby('team').margin.transform(lambda x: x.rolling(ROLLING_AVE_WINDOW).mean().shift())

    # Join back to original df
    random_df = (pd.merge(random_df, 
                         long_df[['GAME_ID', 'team', 'ave_margin']], 
                         left_on=['GAME_ID', 'team_1'],
                         right_on=['GAME_ID', 'team']
                        )
                 .rename(columns={'ave_margin': 'ave_margin_1'})
                 .drop(columns=['team'])
                )

    random_df = (pd.merge(random_df, 
                         long_df[['GAME_ID', 'team', 'ave_margin']], 
                         left_on=['GAME_ID', 'team_2'],
                         right_on=['GAME_ID', 'team']
                        )
                 .rename(columns={'ave_margin': 'ave_margin_2'})
                 .drop(columns=['team'])
                )
    random_df['ave_margin_diff'] = random_df['ave_margin_1'] - random_df['ave_margin_2']
    
    random_df['winner_selection'] = np.select(
        [
            random_df.margin > 0,
            random_df.margin == 0,
            random_df.margin < 0
        ],
        [
            'team_1',
            'draw',
            'team_2'
        ],
        'unknown'
    )
    
    
    # Drop null rows due to rolling mean
    random_df = random_df[~(random_df.ave_margin_1.isnull()) & ~(random_df.ave_margin_2.isnull())].reset_index(drop=True)
    return random_df, elos_dict, trueskill_ratings_dict

def get_ave_margin_dict(historic_df, window=20):
    '''This function returns a dict of ave margins over a certain window so it can be mapped to this weeks games'''    
    
    # Split df into long
    df1 = historic_df[['GAME_ID', 'date', 'team_1', 'margin']].rename(columns={'team_1': 'team'})
    df2 = historic_df[['GAME_ID', 'date', 'team_2', 'margin']].rename(columns={'team_2': 'team'}).assign(margin=lambda df: df.margin*-1)
    long_df = pd.concat([df1, df2], sort=False).sort_values(by='GAME_ID').reset_index(drop=True)
    long_df['ma'] = long_df.groupby('team').margin.transform(lambda x: x.rolling(window).mean())
    ma_dict = long_df.groupby('team').ma.last().to_dict()
    return ma_dict

def make_todays_feature_set(elo_ratings_dict, trueskill_ratings_dict, ave_margin_dict):
    username = os.environ.get('BF_USERNAME_JAMES')
    pw = os.environ.get('BF_PW_JAMES')
    app_key = os.environ.get('BF_APP_KEY_JAMES')

    trading = APIClient(username, pw, app_key=app_key, lightweight=True)
    trading.login_interactive()
    
    nrl_competition_id = '10564377'

    # Define a market filter
    event_filter = betfairlightweight.filters.market_filter(
        market_type_codes=['MATCH_ODDS'],
        competition_ids=[nrl_competition_id]
    )
    
    # Get upcoming events from bf api
    event_info = get_upcoming_event_info(trading, event_filter)
    
    ###########
    # Map elo
    ###########
    event_info['elo_1'] = event_info.team_1.map(elo_ratings_dict)
    event_info['elo_2'] = event_info.team_2.map(elo_ratings_dict)    
    event_info['elo_prob_1'] = 1 / (1 + 10**((event_info.elo_2 - event_info.elo_1) / 400))
    event_info['elo_prob_2'] = 1 - event_info['elo_prob_1']
    event_info['elo_odds_1'] = 1 / event_info['elo_prob_1']
    event_info['elo_odds_2'] = 1 / event_info['elo_prob_2']
    
    ##########
    # Map trueskill
    ##########
    event_info['trueskill_mu_1'] = event_info.team_1.map(lambda x: trueskill_ratings_dict[x].mu)
    event_info['trueskill_mu_2'] = event_info.team_2.map(lambda x: trueskill_ratings_dict[x].mu)
    event_info['trueskill_sigma_1'] = event_info.team_1.map(lambda x: trueskill_ratings_dict[x].sigma)
    event_info['trueskill_sigma_2'] = event_info.team_2.map(lambda x: trueskill_ratings_dict[x].sigma)
    
    ##########
    # Map margin
    ##########
    event_info['ave_margin_1'] = event_info.team_1.map(ave_margin_dict)
    event_info['ave_margin_2'] = event_info.team_2.map(ave_margin_dict)

    event_info = event_info.assign(localMarketStartTime=lambda df: pd.to_datetime(df.market_start_time).apply(lambda row: 
                                                                                 (row.replace(tzinfo=pytz.utc)
                                                                                     .astimezone(pytz.timezone('Australia/Melbourne'))
                                                                                     .strftime("%a %B %e, %I:%M%p"))))
    return event_info


def get_upcoming_event_info(client, market_filter):
    market_catalogue = client.betting.list_market_catalogue(market_filter,
                                     market_projection=["RUNNER_DESCRIPTION","COMPETITION","EVENT","EVENT_TYPE", 'MARKET_START_TIME'],
                                     max_results=500)
    
    selection_df1 = (pd.DataFrame([cat['runners'][0] for cat in market_catalogue])[['runnerName', 'selectionId']]
                    .rename(columns={
                        'runnerName': 'team_1',
                        'selectionId': 'selection_id_1'
                    })
                    )
    selection_df2 = (pd.DataFrame([cat['runners'][1] for cat in market_catalogue])[['runnerName', 'selectionId']]
                    .rename(columns={
                        'runnerName': 'team_2',
                        'selectionId': 'selection_id_2'
                    })
                    )
    
    
    event_df = (pd.DataFrame([cat['event'] for cat in market_catalogue])
               .rename(columns={
                   'id': 'event_id',
                   'name': 'event_name',
               })
               )
    
    event_df['market_id'] = [str(cat['marketId']) for cat in market_catalogue]
    event_df['market_start_time'] = [str(cat['marketStartTime']) for cat in market_catalogue]
    
    competition_df = (pd.DataFrame([cat['competition'] for cat in market_catalogue])
               .rename(columns={
                   'id': 'competition_id',
                   'name': 'competition_name',
               })
               )
    
    return pd.concat([selection_df1, selection_df2, event_df, competition_df], axis=1)

def get_newest_model():
    # Find newest model
    model_paths = [path for path in os.listdir('models/') if 'automl_model' in path]
    model_dates = [model_path.split('automl_model_')[1] for model_path in model_paths]
    
    # Check that there is a model in the path
    assert len(model_paths) > 0
    
    # Get most recent model date and the path to the most recent model
    newest_model_date = max(model_dates)
    newest_model_idx = model_dates.index(newest_model_date)
    model_path_to_load = 'models/' + model_paths[newest_model_idx]
    
    # Load newest model
    aml = h2o.load_model(model_path_to_load)
    return aml

def save_to_local_machine(path, json_file):
    with open(path, "w") as f:
        json.dump(json_file, f)
