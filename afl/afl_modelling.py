'''
This script will allow you to modelling our AFL data to create predictions. We will train a variety of models, 
then tune our hyperparameters so that we are ready to make week by week predictions.
'''

# Import libraries
from afl_feature_creation import prepare_afl_features
import afl_feature_creation
import afl_data_cleaning
import datetime
import pandas as pd
import numpy as np
from afl_data_cleaning import prepare_afl_data
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import RFECV
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn import feature_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Define a function which splits our DataFrame so each game is on one row instead of two
def get_df_on_one_line(df):
    cols_to_drop = ['Team', 'Date', 'home_team', 'away_team',
       'Home?', 'Opposition', 
       'Opposition Behinds', 'Opposition Goals', 'Opposition Points', 'Points',
       'Round', 'Venue', 'Season', 'Status']
    
    home_df = df[df['Home?'] == 1]
    away_df = df[df['Home?'] == 0]
    away_df = away_df.drop(columns=cols_to_drop)
    home_df = home_df.drop(columns=['Team', 'Home?'])

    # Rename away_df columns
    away_df_renamed = away_df.rename(columns={col: col + '_away' for col in away_df.columns if col != 'Game'})
    merged_df = pd.merge(home_df, away_df_renamed, on='Game')
    return merged_df

def get_diff_df(df):
	# Create a DataFrame with the difference between features - let's call it diff_df
	# Create a list of columns which we are subtracting from each other
	cols = [col for col in df.columns if col + '_away' in df.columns and col != 'odds']

	diff_df = df.copy()
	diff_cols = [col + '_diff' for col in cols]

	for col in cols:
	    diff_df[col + '_diff'] = df[col] - df[col + '_away']
	    diff_df = diff_df.drop(columns=[col, col + '_away'])
	    
	# Create an implied odds feature
	diff_df['implied_odds_prob'] = 1 / diff_df['odds']
	diff_df['implied_odds_prob_away'] = 1 / diff_df['odds_away']
	return diff_df

# Define a function which implements our xgb stacker
def implement_xgb_stacking(train_x, train_y, test_x, classifier_list):
    # Get half of the DataFrame to train the base learners on and the other half to create base predictions
    xtrain_base, xpred_base, ytrain_base, ypred_base = train_test_split(train_x, train_y, test_size=0.5, random_state=42)
    
    # Create two DataFrames - one for our base predictions to train the meta-learner and the other to be used 
    # for the meta-learner's predictions
    base_df = pd.DataFrame()
    test_base_df = pd.DataFrame()
    
    # Loop over the classifiers to train the base-learners
    for clf in classifier_list:
        # Fit each classifier using the base training data
        clf.fit(xtrain_base, ytrain_base)
        
        # Create a base predictions DataFrame to be used to train the meta-learner
        base_preds = clf.predict(xpred_base)
        base_df[clf.__class__.__name__] = base_preds
        
        # Use the base learners to create test_predictions to be used for our final meta-learner predictions
        test_preds = clf.predict(test_x)
        test_base_df[clf.__class__.__name__] = test_preds

    meta_clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0.25, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
    
    # Train the meta-learner
    meta_clf.fit(base_df, ypred_base)
    
    # Predict
    final_preds = meta_clf.predict(test_base_df)
    return final_preds

# Hard code estimators so we don't have to use cross val again
all_estimators = [LogisticRegression(C=0.75, class_weight=None, dual=False, fit_intercept=True,
       intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
       penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
       verbose=0, warm_start=False),
RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
     max_iter=None, normalize=False, random_state=None, solver='auto',
     tol=0.001),
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
         max_depth=15, max_features='sqrt', max_leaf_nodes=None,
         min_impurity_decrease=0.0, min_impurity_split=None,
         min_samples_leaf=1, min_samples_split=10,
         min_weight_fraction_leaf=0.0, n_estimators=750, n_jobs=-1,
         oob_score=False, random_state=5, verbose=0, warm_start=False),
GaussianNB(priors=None),
LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
           solver='svd', store_covariance=False, tol=0.0001)]

# Hard code best cols
best_cols = ['home_elo',
'away_elo',
'GA_ave_6_diff',
'CP_ave_6_diff',
'UP_ave_6_diff',
'ED_ave_6_diff',
'CM_ave_6_diff',
'MI5_ave_6_diff',
'One.Percenters_ave_6_diff',
'BO_ave_6_diff',
'HB_ave_6_diff',
'M_ave_6_diff',
'G_ave_6_diff',
'T_ave_6_diff',
'HO_ave_6_diff',
'I50_ave_6_diff',
'CL_ave_6_diff',
'CG_ave_6_diff',
'R50_ave_6_diff',
'FF_ave_6_diff',
'FA_ave_6_diff',
'AF_ave_6_diff',
'SC_ave_6_diff',
'disposal_efficiency_ave_6_diff',
'R50_efficiency_ave_6_diff',
'I50_efficiency_ave_6_diff',
'Adj_elo_ave_margin_ave_6_diff',
'average_elo_opponents_beaten_6_diff',
'average_elo_opponents_lost_6_diff',
'Margin_ave_6_diff',
'implied_odds_prob',
'implied_odds_prob_away']

def get_fixture(path):
	# Get the afl fixture
	fixture = pd.read_csv(path)

	# Replace team names and reformat
	fixture = fixture.replace({'Brisbane Lions': 'Brisbane', 'Footscray': 'Western Bulldogs'})
	fixture['Date'] = pd.to_datetime(fixture['Date']).dt.date.astype(str)
	fixture = fixture.rename(columns={"Home.Team": "home_team", "Away.Team": "away_team"})
	return fixture

def get_next_week_odds(path):
	# Get next week's odds
	next_week_odds = pd.read_csv(path)
	next_week_odds = next_week_odds.rename(columns={"team_1": "home_team", 
                                                "team_2": "away_team", 
                                                "team_1_odds": "odds", 
                                                "team_2_odds": "odds_away"
                                               })
	return next_week_odds

def create_next_weeks_game_ids(afl_data):
	odds = get_next_week_odds("data/weekly_odds.csv")

	# Get last week's Game ID
	last_afl_data_game = afl_data['Game'].iloc[-1]

	# Create Game IDs for next week
	game_ids = [(i+1) + last_afl_data_game for i in range(odds.shape[0])]
	return game_ids

def get_next_week_df(afl_data):
	# Get the fixture and the odds for next week's footy games
	fixture = get_fixture("data/afl_fixture_2018.csv")
	next_week_odds = get_next_week_odds("data/weekly_odds.csv")
	next_week_odds['Game'] = create_next_weeks_game_ids(afl_data)

	# Get today's date and next week's date and create a DataFrame for next week's games
	todays_date = datetime.datetime.today().strftime('%Y-%m-%d')
	date_in_7_days = (datetime.datetime.today() + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
	fixture = fixture[(fixture['Date'] >= todays_date) & (fixture['Date'] < date_in_7_days)].drop(columns=['Season', 'Season.Game', 'Round'])
	next_week_df = pd.merge(fixture, next_week_odds, on=['home_team', 'away_team'])

	# Split the DataFrame onto two rows for each game
	h_df = next_week_df[['Date', 'Game', 'home_team', 'away_team', 'odds']]
	h_df['Team'] = h_df['home_team']
	h_df['Opposition'] = h_df['away_team']
	h_df['Home?'] = 1
	a_df = next_week_df[['Date', 'Game', 'home_team', 'away_team', 'odds_away']].rename(columns={'odds_away': 'odds'})
	a_df['Team'] = a_df['away_team']
	a_df['Opposition'] = a_df['home_team']
	a_df['Home?'] = 0
	next_week = a_df.append(h_df).sort_values(by='Game')
	return next_week

def prepare_afl_prediction_set():
	# Get our original dataset
	afl_data = prepare_afl_data()
	afl_data_original_copy = afl_data.copy()
	ordered_cols = afl_data.columns
	
	# Get next week's DataFrame so that we can create our features for the prediction set
	next_week_df = get_next_week_df(afl_data)

	# Append next week's games to our afl_data DataFrame
	afl_data = afl_data.append(next_week_df).reset_index(drop=True)
	afl_data = afl_data[ordered_cols]

	# Create disposal efficiency column
	afl_data['disposal_efficiency'] = afl_data['ED'] / afl_data['D']

	# Create rolling averages for our statistics
	cols_indx_start = afl_data.columns.get_loc("GA")
	afl_avgs = afl_feature_creation.create_rolling_averages(afl_data, 6, afl_data.columns[cols_indx_start:])

	# Create form between teams feature
	afl_avgs = afl_feature_creation.form_between_teams(afl_avgs, 6)

	# Apply elos
	elos, probs, elo_dict = afl_feature_creation.elo_applier(afl_data, 24)
	afl_avgs['home_elo'] = afl_avgs['Game'].map(elos).str[0]
	afl_avgs['away_elo'] = afl_avgs['Game'].map(elos).str[1]

	# Get elos for next week
	afl_avgs.loc[afl_avgs['home_elo'].isna(), 'home_elo'] = afl_avgs[afl_avgs['home_elo'].isna()]['home_team'].map(elo_dict)
	afl_avgs.loc[afl_avgs['away_elo'].isna(), 'away_elo'] = afl_avgs[afl_avgs['away_elo'].isna()]['away_team'].map(elo_dict)

	# Create Adjusted Margin and then Average it over a 6 game window
	afl_avgs = afl_feature_creation.map_elos(afl_avgs)
	afl_avgs['Adj_elo_ave_margin'] = afl_avgs['Margin'] * afl_avgs['elo_Opp'] / afl_avgs['elo']
	afl_avgs = afl_feature_creation.create_rolling_averages(afl_avgs, 6, ['Adj_elo_ave_margin'])

	# Create average elo of opponents beaten/lost to feature
	afl_avgs = afl_feature_creation.create_ave_elo_opponent(afl_avgs, 6, beaten_or_lost='beaten')
	afl_avgs = afl_feature_creation.create_ave_elo_opponent(afl_avgs, 6, beaten_or_lost='lost')

	# Create regular margin rolling average
	afl_avgs = afl_feature_creation.create_rolling_averages(afl_avgs, 6, ['Margin'])

	# Get each footy match on individual rows and drop irrelevant columns
	one_line = get_df_on_one_line(afl_avgs)

	# Create a differential DataFrame
	diff_df = get_diff_df(one_line)
	diff_df = diff_df.drop(columns=['odds', 'odds_away', 'Round', 'Season']).select_dtypes(include=[np.number])

	game_ids_next_round = create_next_weeks_game_ids(afl_data_original_copy)

	prediction_feature_set = diff_df[diff_df['Game'].isin(game_ids_next_round)].dropna(axis=1)

	return prediction_feature_set

def prepare_afl_predictions_df(estimator_list=all_estimators, best_cols=best_cols, window=6, k_factor=24):
	feature_df = prepare_afl_features(window, k_factor)
	afl_with_NaNs = get_df_on_one_line(feature_df)
	# Drop NA rows which were a result of calculating rolling averages
	afl = afl_with_NaNs.dropna().sort_values(by='Game')
	# Get a differential DataFrame - subtracting the away features from the home features
	
	# Drop columns which leak data and which we don't need
	dropped_cols = ['Behinds', 'Goals', 'Opposition Behinds', 'Opposition Goals', 'Opposition Points', 'Points',
	               'Behinds_away', 'Goals_away', 'home_win_away', 'home_elo_away', 'away_elo_away', 'elo', 'elo_Opp', 'CCL_ave_6', 
	                'SCL_ave_6', 'SI_ave_6', 'MG_ave_6', 'TO_ave_6', 'ITC_ave_6', 'T5_ave_6', 'CCL_ave_6_away', 'SCL_ave_6_away',
	               'SI_ave_6_away', 'MG_ave_6_away', 'TO_ave_6_away', 'ITC_ave_6_away', 'T5_ave_6_away', 'elo_away', 'elo_Opp_away']
	afl = afl.drop(columns=dropped_cols)

	diff_df = get_diff_df(afl)

	# Create our train set from our afl DataFrame; drop the columns which leak the result, duplicates, and the advanced
	# stats which don't have data until 2015

	# Create our train sets
	X = diff_df.drop(columns=['home_win', 'odds', 'odds_away']).select_dtypes(include=[np.number])
	y = diff_df['home_win']

	next_round_features = prepare_afl_prediction_set()

	features = next_round_features.drop(columns=['CCL_ave_6_diff', 'SCL_ave_6_diff', 'SI_ave_6_diff', 'MG_ave_6_diff', 'TO_ave_6_diff', 'ITC_ave_6_diff', 'T5_ave_6_diff', \
		'home_elo_diff', 'away_elo_diff', 'elo_diff', 'elo_Opp_diff']).columns

	preds_next_round = implement_xgb_stacking(X[features].drop(columns='Game'), y, next_round_features[features].drop(columns='Game'), all_estimators)
	
	next_week_df = get_next_week_df(afl)

	preds_df = pd.DataFrame({
		"Game": next_round_features['Game'],
		"Prediction (home_win)": preds_next_round
		})

	final_df = pd.merge(preds_df, next_week_df[['home_team', 'away_team', 'Game']], on='Game')
	final_df = pd.merge(final_df, next_round_features[features], on='Game')
	final_df = final_df.drop_duplicates()
	final_df['home_odds'] = 1 / final_df['implied_odds_prob']
	final_df['away_odds'] = 1 / final_df['implied_odds_prob_away']
	return final_df

if __name__ == "__main__":
	prepare_afl_predictions_df()