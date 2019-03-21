import functions.prod_functions as prod_functions
import pandas as pd
import json
import h2o
from h2o.automl import H2OAutoML
import betfairlightweight
from betfairlightweight import APIClient
from betfairlightweight import filters
import pytz
import datetime
import boto3
import json
import numpy
import logging
import numpy as np

def main():
	# Check which round we predicted last week
	print("Geting last round predicted")
	with open('last_round_predicted.txt', 'r+') as f:
		last_round = int(f.read())
		ROUND = str(last_round + 1)

	print(f"Last round predicted is {last_round}; let's predict {ROUND}")

	print("Get train df and elo, trueskill and ave margin dicts")
	train, elos_dict, trueskill_ratings_dict = prod_functions.get_train_features()

	ave_margin_dict = prod_functions.get_ave_margin_dict(train)

	print("Make todays feature set by connecting to Betfair's API and mapping the dicts to the teams playing")
	todays_features_with_info = prod_functions.make_todays_feature_set(elos_dict, trueskill_ratings_dict, ave_margin_dict)

	feature_cols = ['elo_prob_1', 'elo_prob_2', 'trueskill_mu_1', 'trueskill_mu_2', 'trueskill_sigma_1', 'trueskill_sigma_2',
	 'ave_margin_1', 'ave_margin_2']


	print("Init h2o")
	h2o.init()

	print("Get newest model")
	aml = prod_functions.get_newest_model()

	todays_features_h2o = h2o.H2OFrame(todays_features_with_info[feature_cols])

	print("Predict upcoming matches")
	preds = aml.predict(todays_features_h2o).as_data_frame()

	todays_features_with_info['model_prob_1'] = preds.team_1.tolist()
	todays_features_with_info['model_odds_1'] = 1 / todays_features_with_info['model_prob_1']

	todays_features_with_info['model_prob_2'] = preds.team_2.tolist()
	todays_features_with_info['model_odds_2'] = 1 / todays_features_with_info['model_prob_2']
	todays_features_with_info['predicted_winner'] = np.where(
		todays_features_with_info.model_odds_1 < todays_features_with_info.model_odds_2, 
		todays_features_with_info.team_1, 
		todays_features_with_info.team_2
		)

	print("Create json predictions")
	todays_json = prod_functions.create_json_predictions(todays_features_with_info, ROUND)

	# Get todays date
	todays_date = datetime.datetime.now().strftime('%Y%m%d')

	print("Save to local machine")
	prod_functions.save_to_local_machine(f'predictions/super_rugby_{todays_date}.json', todays_json)
	todays_features_with_info.to_csv(f'predictions/super_rugby_{todays_date}.csv', index=False)

	print("Writing the last round we predicted to txt file")
	with open('last_round_predicted.txt', 'w') as f:
		f.write(str(ROUND))

	h2o.cluster().shutdown()

if __name__ == "__main__":
	main()
