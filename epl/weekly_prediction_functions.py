import pandas as pd
import datetime
import imp
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pytz


def create_game_info_df(game_info_path):
    game_info = (pd.read_csv(game_info_path,
                        dtype={
                            'awaySelectionId': str,
                            'drawSelectionId': str,
                            'homeSelectionId': str,
                            'marketId': str,
                            'eventId': str,
                            'competitionId': str
                        })
                .assign(Date=lambda df: pd.to_datetime(df.marketStartTime).dt.normalize(),
                        marketStartTime=lambda df: pd.to_datetime(df.marketStartTime))
                .assign(localMarketStartTime=lambda df: df.marketStartTime.apply(lambda row: 
                                                                                 (row.replace(tzinfo=pytz.utc)
                                                                                     .astimezone(pytz.timezone('Australia/Melbourne'))
                                                                                     .strftime("%a %B %e, %I:%M%p"))))
                .replace({
                    'Man Utd': 'Man United',
                    'C Palace': 'Crystal Palace'
                    }))
    return game_info


# Define a function which creates a dataframe of the fixture for the 2018/19 season
def create_fixture_df(fixture_path):
    mapping = {
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Tottenham Hotspur': 'Tottenham',
    'Wolverhampton': 'Wolves'
}
    fixture = (pd.read_csv(fixture_path)
                  .dropna()
                  .query("Date != 'Date'")
                  .assign(Date=lambda df: df.Date.astype(str) + ' ' + df.Year.astype(str))
                  .assign(Date=lambda df: pd.to_datetime(df.Date),
                          round=[i for i in range(1, 39) for j in range(10)],
                          season='1819')
                  .reset_index(drop=True)
                  .rename(columns={'Home': 'HomeTeam', 'Away': 'AwayTeam'})
                  .replace(mapping))
    return fixture


def create_predictions_df(train_x, train_y, test_x, production_df, game_info_df):
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    predictions = lr.predict_proba(test_x)
    odds = 1 / predictions
    merged_cols = ['Date', 'HomeTeam', 'AwayTeam', 'gameId', 'homeModelledOdds', 'drawModelledOdds', 'awayModelledOdds', 'eloHome', 'eloAway']
    print(predictions)
    print(game_info_df)
    print(production_df)
    # Assign the modelled odds to our predictions df
    predictions_df = (production_df.assign(homeModelledOdds=[i[2] for i in (1 / predictions)],
                                          drawModelledOdds=[i[1] for i in (1 / predictions)],
                                          awayModelledOdds=[i[0] for i in (1 / predictions)])
                                   .pipe(lambda df: pd.merge(df[merged_cols], game_info_df.drop(columns='Date'), on=['HomeTeam', 'AwayTeam'])))
    
    print(predictions_df)
    # Assign the modelled winner to our predictions df
    predictions_df = (predictions_df.assign(favourite=lambda df: df[['homeModelledOdds', 'drawModelledOdds', 'awayModelledOdds']].idxmin(axis=1))
                                    .assign(modelledWinner=lambda df: df.apply(lambda row:
                                                         row.HomeTeam if row.favourite == 'homeModelledOdds' else (
                                                         row.AwayTeam if row.favourite == 'awayModelledOdds' else 'The Draw'), axis=1)))
    
    
    return predictions_df