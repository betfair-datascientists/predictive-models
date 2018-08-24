import pandas as pd
from sklearn.metrics import log_loss
import math
import numpy as np
from functools import reduce

# Define a function which
def create_df(path):
		df = (pd.read_csv(path, dtype={'season': str})
						.assign(Date=lambda df: pd.to_datetime(df.Date))
						.pipe(lambda df: df.dropna(thresh=len(df) - 2, axis=1))  # Drop cols with NAs
						.dropna(axis=0)  # Drop rows with NAs
						.sort_values('Date')
						.reset_index(drop=True)
						.assign(gameId=lambda df: list(df.index + 1),
										Year=lambda df: df.Date.apply(lambda row: row.year),
									homeWin=lambda df: df.apply(lambda row: 1 if row.FTHG > row.FTAG else 0, axis=1),
									awayWin=lambda df: df.apply(lambda row: 1 if row.FTAG > row.FTHG else 0, axis=1),
									result=lambda df: df.apply(lambda row: 'home' if row.FTHG > row.FTAG else ('draw' if row.FTHG == row.FTAG else 'away'), axis=1)))
		return df


def create_stats_df(path):
		df = (pd.read_csv(path, dtype={'season': str})
						.assign(Date=lambda df: pd.to_datetime(df.Date))
						.pipe(lambda df: df.dropna(thresh=len(df) - 2, axis=1))  # Drop cols with NAs
						.dropna(axis=0)  # Drop rows with NAs
						.sort_values('Date')
						.reset_index(drop=True)
						.assign(gameId=lambda df: list(df.index + 1)))
		
		stats_cols = ['gameId', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 
									'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
		
		stats = df[stats_cols].copy()
		return stats


def create_betting_df(path):
		df = (pd.read_csv(path, dtype={'season': str})
						.assign(Date=lambda df: pd.to_datetime(df.Date))
						.pipe(lambda df: df.dropna(thresh=len(df) - 2, axis=1))  # Drop cols with NAs
						.dropna(axis=0)  # Drop rows with NAs
						.sort_values('Date')
						.reset_index(drop=True)
						.assign(gameId=lambda df: list(df.index + 1),
										Year=lambda df: df.Date.apply(lambda row: row.year),
										homeWin=lambda df: df.apply(lambda row: 1 if row.FTHG > row.FTAG else 0, axis=1),
										awayWin=lambda df: df.apply(lambda row: 1 if row.FTAG > row.FTHG else 0, axis=1),
										result=lambda df: df.apply(lambda row: 'home' if row.FTHG > row.FTAG else ('draw' if row.FTHG == row.FTAG else 'away'), axis=1)))
		
		betting_cols = ['B365A', 'B365D', 'B365H', 'BWA', 'BWD', 'BWH', 'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 
										'BbAv>2.5', 'BbAvA', 'BbAvAHA', 'BbAvAHH', 'BbAvD', 'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 
										'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU', 'Day', 'Div', 'IWA', 'IWD', 
										'IWH', 'LBA', 'LBD', 'LBH', 'Month', 'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH', 'Year', 
										'homeWin', 'awayWin', 'result', 'HomeTeam', 'AwayTeam', 'gameId']
		
		betting = df[betting_cols].copy()
		
		return betting


def create_historic_games_df(path):
		historic_games = (pd.read_csv(path)
													.assign(Date=lambda df: pd.to_datetime(df.Date),
																	gameId=-1,
																	homeWin=lambda df:
																	df.apply(lambda row: 1 if row.FTHG > row.FTAG else 0, axis='columns')))
		return historic_games


def create_team_info_df(path):
		df = create_df(path)
		team_info = df[['gameId', 'Date', 'season', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee']].copy()
		return team_info


def create_all_games_df(epl_data_path, historic_games_path):
		df = create_df(epl_data_path)
		historic_games = create_historic_games_df(historic_games_path)
		
		all_games = (historic_games.append(df[historic_games.columns])
								 .reset_index(drop=True)
								 .assign(awayWin=lambda df: df.apply(lambda row: 1 if row.FTAG > row.FTHG else 0, axis='columns'))
								 .pipe(lambda df: win_pc(df, 5, "HomeTeam", "homeWin", "homeWinPc5"))
								 .pipe(lambda df: win_pc(df, 38, "HomeTeam", "homeWin", "homeWinPc38"))
								 .pipe(lambda df: win_pc(df, 5, "AwayTeam", "awayWin", "awayWinPc5"))
								 .pipe(lambda df: win_pc(df, 38, "AwayTeam", "awayWin", "awayWinPc38"))
								 .assign(gameIdHistoric=lambda df: list(range(1, len(df) + 1))))
		return all_games


def create_multiline_df_all_games(old_all_games_df):
		home_df = (old_all_games_df[old_all_games_df.drop(columns='AwayTeam').columns]
							 .rename(columns={
								'HomeTeam': 'team',
								'FTHG': 'goalsFor',
								'FTAG': 'goalsAgainst',
								'homeElo': 'eloFor',
								'awayElo': 'eloAgainst'})
							 .assign(homeGame=1))
		
		away_df = (old_all_games_df[old_all_games_df.drop(columns='HomeTeam').columns]  # Create an away dataframe
							 .rename(columns={
								'AwayTeam': 'team',
								'FTHG': 'goalsAgainst',
								'FTAG': 'goalsFor',
								'homeElo': 'eloAgainst',
								'awayElo': 'eloFor'
							 })
							 .assign(homeGame=0))
		
		multi_line_df = home_df.append(away_df, sort=True).sort_values(by='gameIdHistoric').reset_index(drop=True)
		return multi_line_df


def create_multiline_df_stats(old_stats_df):
		# Get stats df onto one line and then find rolling averages over a 7 game period
		home_stats_cols = ['HomeTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY',
											 'HR', 'AR']
		away_stats_cols = ['AwayTeam', 'FTAG', 'FTHG', 'HTAG', 'HTHG', 'AS', 'HS', 'AST', 'HST', 'AF', 'HF', 'AC', 'HC', 'AY', 'HY',
											 'AR', 'HR']
		stats_cols_mapping = ['team', 'goalsFor', 'goalsAgainst', 'halfTimeGoalsFor', 'halfTimeGoalsAgainst', 'shotsFor',
													'shotsAgainst', 'shotsOnTargetFor', 'shotsOnTargetAgainst', 'freesFor', 'freesAgainst', 
													'cornersFor', 'cornersAgainst', 'yellowsFor', 'yellowsAgainst', 'redsFor', 'redsAgainst']
		home_mapping = {old_col: new_col for old_col, new_col in zip(home_stats_cols, stats_cols_mapping)}
		away_mapping = {old_col: new_col for old_col, new_col in zip(away_stats_cols, stats_cols_mapping)}
		
		# Put each team onto an individual row
		multi_line_stats = (old_stats_df[['gameId'] + home_stats_cols]
												.rename(columns=home_mapping)
												.append((old_stats_df[['gameId'] + away_stats_cols])
																.rename(columns=away_mapping), sort=True)
												.sort_values(by='gameId')
												.reset_index(drop=True))
		return multi_line_stats


def create_multiline_df_betting(betting):
		home_col_mapping = {
				'BbAHh': 'sizeOfHandicap',
				'BbAvAHH': 'avAsianHandicapOddsFor',
				'BbAvAHA': 'avAsianHandicapOddsAgainst',
				'BbAv<2.5': 'avlessthan2.5',
				'BbAv>2.5': 'avgreaterthan2.5'
		}
		
		away_col_mapping = {
				'BbAHh': 'sizeOfHandicap',
				'BbAvAHH': 'avAsianHandicapOddsAgainst',
				'BbAvAHA': 'avAsianHandicapOddsFor',
				'BbAv<2.5': 'avlessthan2.5',
				'BbAv>2.5': 'avgreaterthan2.5'
		}
		
		multi_line_odds = (betting[['gameId', 'HomeTeam', 'BbAHh', 'BbAvAHH', 'BbAvAHA', 'BbAv<2.5', 'BbAv>2.5']]
											 .rename(columns=home_col_mapping)
											 .rename(columns={'HomeTeam': 'team'})
											 .append(betting[['gameId', 'AwayTeam', 'BbAHh', 'BbAvAHH', 'BbAvAHA', 'BbAv<2.5', 'BbAv>2.5']]
															 .rename(columns=away_col_mapping)
															 .rename(columns={'AwayTeam': 'team'})
															 .assign(sizeOfHandicap=lambda df: df.sizeOfHandicap * -1), sort=True)
											 .sort_values(by='gameId')
											 .reset_index(drop=True))
		return multi_line_odds


def create_all_games_features(all_games):
		# Get elos for post 2005
		elos, elo_probs, all_elos_current = elo_applier(df=all_games,
															k_factor=25,
															historic_elos={team: 1500 for team in all_games['HomeTeam'].unique()},
															soft_reset_factor=0.96,
															game_id_col_name="gameIdHistoric")

		# Add elos to the all_games df
		all_games_with_elos = (all_games.assign(homeElo=lambda df: df.gameIdHistoric.map(elos).str[0],
																						awayElo=lambda df: df.gameIdHistoric.map(elos).str[1]))

		# Put the all_games df onto 2 rows per game (with each team's stats on each row)
		multi_line_all_games = create_multiline_df_all_games(all_games_with_elos)

		# Add a elo weighted goals against feature
		multi_line_all_games['wtEloGoalsFor'] = [
				wt_goals_elo(df=multi_line_all_games,
										 game_id_row=row[multi_line_all_games.columns.get_loc('gameIdHistoric') + 1], # Get the index of 'gameIdHistoric'
										 team_row=row[multi_line_all_games.columns.get_loc('team') + 1], # Get the index of 'gameIdHistoric'
										 goalsForOrAgainstCol='goalsFor') for row in multi_line_all_games.itertuples()
		]

		multi_line_all_games['wtEloGoalsAgainst'] = [
				wt_goals_elo(df=multi_line_all_games,
										 game_id_row=row[multi_line_all_games.columns.get_loc('gameIdHistoric') + 1], # Get the index of 'gameIdHistoric'
										 team_row=row[multi_line_all_games.columns.get_loc('team') + 1], # Get the index of 'gameIdHistoric'
										 goalsForOrAgainstCol='goalsAgainst') for row in multi_line_all_games.itertuples()
		]
		
		return multi_line_all_games


def create_stats_features_ema(stats, span):
		multi_line_stats = create_multiline_df_stats(stats)

		ema_features = multi_line_stats[['gameId', 'team']].copy()

		feature_names = multi_line_stats.drop(columns=['gameId', 'team']).columns

		for feature_name in feature_names:
				feature_ema = (multi_line_stats.groupby('team')[feature_name]
											 .transform(lambda row: row.ewm(span=span, min_periods=2)
																	.mean()
																	.shift(1)))
				ema_features[feature_name] = feature_ema
		return ema_features


def create_betting_features_ema(betting, span):
		multi_line_odds = create_multiline_df_betting(betting)

		ema_features = multi_line_odds[['gameId', 'team']].copy()

		feature_names = multi_line_odds.drop(columns=['gameId', 'team']).columns

		for feature_name in feature_names:
				feature_ema = (multi_line_odds.groupby('team')[feature_name]
											 .transform(lambda row: row.ewm(span=span, min_periods=2)
																	.mean()
																	.shift(1)))

				ema_features[feature_name] = feature_ema
		return ema_features


def fill_non_win_pc_na(df, features_multi_line):
		# Find the first year that teams joined the competition
		first_yr_df = df.groupby("HomeTeam").Year.first()

		# Fill homewinPc columns after - we cannot fill them now as the teams aren't separated into home and away columns
		non_win_pc_cols = [col for col in features_multi_line.columns if 'WinPc' not in col]
		win_pc_cols = [col for col in features_multi_line.columns if 'WinPc' in col]

		features_multi_line = (features_multi_line.loc[:, non_win_pc_cols]
													 .iloc[:250]
													 .dropna()
													 .append(features_multi_line[non_win_pc_cols].iloc[250:])
													 .reset_index(drop=True)
													 .assign(Date=lambda df: pd.to_datetime(df.Date))
													 .assign(firstOrSecondYr=lambda df: df.apply(lambda row:
																 1 if first_yr_df[row.team] == row.Date.year
																			and first_yr_df[row.team] != 2005 else
																 (1 if first_yr_df[row.team] + 1 == row.Date.year
																			 and first_yr_df[row.team] != 2005 else 0),
																 axis=1))
													 .pipe(lambda df:
																 df.apply(lambda col: col.fillna(fillna_first_second_yr_multi_line(df, col.name)),
																					axis='rows'))
													 .pipe(pd.merge, features_multi_line[win_pc_cols + ['gameId', 'team']],
																 on=['gameId', 'team']))
		
		return features_multi_line


def put_features_on_one_line(features_multi_line):
		non_features = ['Date', 'awayWin', 'gameId', 'season', 'homeGame', 'team']

		features = (features_multi_line[features_multi_line.homeGame == 1]
				.rename(columns={col: 'f_' + col + 'Home' for col in features_multi_line.columns if col not in non_features})
				.rename(columns={'team': 'HomeTeam'})
				.pipe(pd.merge, (features_multi_line[features_multi_line.homeGame == 0]
														.rename(columns={'team': 'AwayTeam'})
														.rename(columns={col: 'f_' + col + 'Away' for col in features_multi_line.columns 
																						 if col not in non_features})
														.drop(columns={'Date', 'season', 'homeGame'})), on='gameId'))

		return features


# Define a function which reverts the elos to the mean at the end of the season, by a factor
def revert_elos_to_mean(current_elos, soft_reset_factor):
		elos_mean = np.mean(list(current_elos.values()))
		new_elos_dict = {
				team: (team_elo - elos_mean) * soft_reset_factor + elos_mean for team, team_elo in current_elos.items()
		}
		return new_elos_dict


def round_nearest(x):
		return round(round(x / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))


def elo_applier(df, k_factor, historic_elos, soft_reset_factor, game_id_col_name):
		# Initialise a dictionary with default elos for each team
		for team in df['HomeTeam'].unique():
				if team not in historic_elos.keys():
						historic_elos[team] = 1500
		elo_dict = historic_elos.copy()
		elos, elo_probs = {}, {}
		
		last_season = 0
		
		# Loop over the rows in the DataFrame
		for index, row in df.iterrows():
				# Get the current year
				current_season = row['season']
				
				# If it is a new season, soft-reset elos
				if current_season != last_season:
						elo_dict = revert_elos_to_mean(elo_dict, soft_reset_factor)
				# Get the Game ID
				game_id = row[game_id_col_name]
				
				# Get the team and opposition
				home_team = row['HomeTeam']
				away_team = row['AwayTeam']
				
				# Get the team and opposition elo score
				home_team_elo = elo_dict[home_team]
				away_team_elo = elo_dict[away_team]
				
				# Calculated the probability of winning for the team and opposition
				prob_win_home = 1 / (1 + 10**((away_team_elo - home_team_elo) / 400))
				prob_win_away = 1 - prob_win_home
				
				# Add the elos and probabilities our elos dictionary and elo_probs dictionary based on the Game ID
				elos[game_id] = [home_team_elo, away_team_elo]
				elo_probs[game_id] = [prob_win_home, prob_win_away]
				
				margin = row['FTHG'] - row['FTAG']
				
				# Calculate the new elos of each team
				if margin == 1: # Team wins; update both teams' elo
						new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home) * 1
						new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away) * 1
				elif margin == 2:
						new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home) * 1.5
						new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away) * 1.5
				elif margin == 3:
						new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home) * 1.75
						new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away) * 1.75
				elif margin > 3:
						new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home) * 1.75 * (margin - 3) / 8
						new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away) * 1.75 * (margin - 3) / 8
				elif margin == -1: # Away team wins; update both teams' elo
						new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home) * 1
						new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away) * 1
				elif margin == -2: # Away team wins; update both teams' elo
						new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home) * 1.5
						new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away) * 1.5
				elif margin == -3: # Away team wins; update both teams' elo
						new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home) * 1.75
						new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away) * 1.75
				elif margin < -3: # Away team wins; update both teams' elo
						new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home) * 1.75 * (margin - 3) / 8
						new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away) * 1.75 * (margin - 3) / 8
				elif margin == 0: # Drawn game' update both teams' elo
						new_home_team_elo = home_team_elo + k_factor*(0.5 - prob_win_home) * 1
						new_away_team_elo = away_team_elo + k_factor*(0.5 - prob_win_away) * 1
				
				# Update elos in elo dictionary
				elo_dict[home_team] = new_home_team_elo
				elo_dict[away_team] = new_away_team_elo
		
				last_season = current_season
		return elos, elo_probs, elo_dict


# Define a function which calculates the win percentage for a team at home or away over a given window
def win_pc(df, window, home_away_team_col, home_away_win_col, col_name):
		df = df.copy()
		win_pc_col = (df.groupby(home_away_team_col)[home_away_win_col]
										.transform(lambda row: row.rolling(window).mean().shift(1))
										.rename("homeWinPc"))
		df[col_name] = win_pc_col
		return df


# Create a function which creates a dataframe of current market values
# of each EPL team and maps it to the gameId. The input is our main dataframe
def create_market_values_features(df):
		# Team names are different to our main df - automate the team name mapping
		team_name_mapping = {}

		for team_name in df.HomeTeam.unique():
				for team_name_final in pd.read_csv("data/marketValues.csv").TeamName.unique():
						if team_name in team_name_final:
								team_name_mapping[team_name_final] = team_name

		values = (pd.read_csv("data/marketValues.csv")
								.replace(team_name_mapping)
								.replace({
										'Wolverhampton Wanderers': 'Wolves',
										'Queens Park Rangers': 'QPR',
										'Manchester City': 'Man City',
										'Manchester United': 'Man United'})
								.query('marketValue != "-"')
								.assign(marketValueMillions=lambda df: df.marketValue.astype(int) / 1000000))

		def_vals = values[values['Position'] == 'Defender'].groupby(['TeamName', 'Year'], as_index=False)['marketValueMillions'].sum().rename(columns={'marketValueMillions': 'defMktVal'})
		att_vals = values[values['Position'] == 'Forward'].groupby(['TeamName', 'Year'], as_index=False)['marketValueMillions'].sum().rename(columns={'marketValueMillions': 'attMktVal'})
		gk_vals = values[values['Position'] == 'Goalkeeper'].groupby(['TeamName', 'Year'], as_index=False)['marketValueMillions'].sum().rename(columns={'marketValueMillions': 'gkMktVal'})
		total_vals = values[values['Position'] == 'Total:'].groupby(['TeamName', 'Year'], as_index=False)['marketValueMillions'].sum().rename(columns={'marketValueMillions': 'totalMktVal'})
		final_vals = reduce(lambda left, right: pd.merge(left, right, on=['TeamName', 'Year'], how='inner'), [def_vals, att_vals, gk_vals, total_vals])
		final_vals['midMktVal'] = final_vals.totalMktVal - final_vals.defMktVal - final_vals.attMktVal - final_vals.gkMktVal

		mkt_vals_joined = pd.merge(df[['gameId', 'Year', 'HomeTeam', 'AwayTeam']], final_vals.rename(columns={'TeamName': 'HomeTeam'}), on=['HomeTeam', 'Year'], how='left')
		mkt_vals_joined = pd.merge(mkt_vals_joined, final_vals.rename(columns={'TeamName': 'AwayTeam'}), on=['AwayTeam', 'Year'], how='left', suffixes=('H', 'A'))

		for col in ['attMktVal', 'midMktVal', 'defMktVal', 'gkMktVal', 'totalMktVal']:
				mkt_vals_yrlyH = mkt_vals_joined.groupby(['Year', 'HomeTeam'])[col + 'H'].first().groupby(level=0).apply(lambda x: x / x.sum()).unstack()
				mkt_vals_yrlyA = mkt_vals_joined.groupby(['Year', 'AwayTeam'])[col + 'A'].first().groupby(level=0).apply(lambda x: x / x.sum()).unstack()
				mkt_vals_joined[col[:-3] + 'H%'] = mkt_vals_joined.apply(lambda x: mkt_vals_yrlyH.to_dict()[x['HomeTeam']][x['Year']] * 100, axis=1)
				mkt_vals_joined[col[:-3] + 'A%'] = mkt_vals_joined.apply(lambda x: mkt_vals_yrlyA.to_dict()[x['AwayTeam']][x['Year']] * 100, axis=1)
		return mkt_vals_joined


# Create and apply weighted goals for elo function
def wt_goals_elo(df, game_id_row, team_row, goalsForOrAgainstCol):
		wt_goals = (df[(df.gameIdHistoric < game_id_row) & (df.team == team_row)]
									.pipe(lambda df: (df.eloAgainst * df[goalsForOrAgainstCol]).sum() / df.eloAgainst.sum()))
		return wt_goals


def fillna_first_second_yr_multi_line(df, col):
		if df['team'].dtype == 'O':
				return 0
		value = (df.copy()
							 .loc[df['firstOrSecondYr'] == 1, col]
							 .mean())
		return value


def fillna_first_second_yr_single_line_win_pc(df, col):
		if 'homeWinPc' in col:
				value = df.groupby('firstOrSecondYr').homeWin.mean()[1]
		elif 'awayWinPc' in col:
				value = df.groupby('firstOrSecondYr').awayWin.mean()[1]
		else:
				value = -1
		return value


def create_feature_df(df=None):
	##########################
	## Create segmented dfs ##
	##########################
	
	if df is None:
		df = create_df('data/epl_data.csv')

	stats_cols = ['gameId', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 
								'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']

	betting_cols = ['B365A', 'B365D', 'B365H', 'BWA', 'BWD', 'BWH', 'Bb1X2', 'BbAH', 'BbAHh', 'BbAv<2.5', 
							'BbAv>2.5', 'BbAvA', 'BbAvAHA', 'BbAvAHH', 'BbAvD', 'BbAvH', 'BbMx<2.5', 'BbMx>2.5', 
							'BbMxA', 'BbMxAHA', 'BbMxAHH', 'BbMxD', 'BbMxH', 'BbOU', 'Day', 'Div', 'IWA', 'IWD', 
							'IWH', 'LBA', 'LBD', 'LBH', 'Month', 'VCA', 'VCD', 'VCH', 'WHA', 'WHD', 'WHH', 'Year', 
							'homeWin', 'awayWin', 'result', 'HomeTeam', 'AwayTeam', 'gameId']

	team_info = df[['gameId', 'Date', 'season', 'HomeTeam', 'AwayTeam', 'FTR', 'HTR', 'Referee']].copy()                  
	stats = df[stats_cols].copy()
	betting = df[betting_cols].copy()
	historic_games = create_historic_games_df('data/historic_games_pre2005.csv')
		
	all_games = (historic_games.append(df[historic_games.columns])
							 .reset_index(drop=True)
							 .assign(awayWin=lambda df: df.apply(lambda row: 1 if row.FTAG > row.FTHG else 0, axis='columns'))
							 .pipe(lambda df: win_pc(df, 5, "HomeTeam", "homeWin", "homeWinPc5"))
							 .pipe(lambda df: win_pc(df, 38, "HomeTeam", "homeWin", "homeWinPc38"))
							 .pipe(lambda df: win_pc(df, 5, "AwayTeam", "awayWin", "awayWinPc5"))
							 .pipe(lambda df: win_pc(df, 38, "AwayTeam", "awayWin", "awayWinPc38"))
							 .assign(gameIdHistoric=lambda df: list(range(1, len(df) + 1))))

	########################
	## Create feature dfs ##
	########################

	print("Creating all games feature DataFrame")
	features_all_games = create_all_games_features(all_games)

	print("Creating stats feature DataFrame")
	features_stats = create_stats_features_ema(stats, span=49)

	print("Creating odds feature DataFrame")
	features_odds = create_betting_features_ema(betting, span=10)

	print("Creating market values feature DataFrame")
	features_market_values = create_market_values_features(df) # This creates a df with one game per row

	all_games_cols = ['Date', 'gameId', 'team', 'season', 'homeGame', 'homeWinPc38', 'homeWinPc5', 'awayWinPc38', 'awayWinPc5', 'eloFor', 'eloAgainst', 'wtEloGoalsFor', 'wtEloGoalsAgainst']

	features_multi_line = (features_all_games[all_games_cols]
											 .pipe(pd.merge, features_stats, on=['gameId', 'team'])
											 .pipe(pd.merge, features_odds, on=['gameId', 'team']))
	print("Filling NAs")

	# Put each instance on an individual row
	features_with_na = put_features_on_one_line(features_multi_line)

	market_val_feature_names = ['attMktH%', 'attMktA%', 'midMktH%', 'midMktA%', 'defMktH%', 'defMktA%', 'gkMktH%', 'gkMktA%', 'totalMktH%', 'totalMktA%']
	print("Merging stats, odds and market values into one features DataFrame")

	# Merge our team values dataframe to features and result from df
	features_with_na = (features_with_na.pipe(pd.merge, features_market_values[market_val_feature_names + ['gameId']].rename(columns={col: 'f_' + col for col in market_val_feature_names}), on='gameId')
										.pipe(pd.merge, df[['HomeTeam', 'AwayTeam', 'gameId', 'result', 'B365A', 'B365D', 'B365H']], on=['HomeTeam', 'AwayTeam', 'gameId'])
										.drop(columns='homeGame'))

	# Drop NAs from calculating the rolling averages - don't drop Win Pc 38 and Win Pc 5 columns
	features = features_with_na.dropna(subset=features_with_na.drop(columns=[col for col in features_with_na.columns if 'WinPc' in col]).columns)

	# Fill NAs for the Win Pc columns
	features = features.fillna(features.mean())

	features = features.rename(columns={
		'B365H': 'f_homeOdds',
		'B365A': 'f_awayOdds',
		'B365D': 'f_drawOdds'
		})

	print("Complete.")
	return features