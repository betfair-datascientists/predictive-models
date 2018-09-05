###################################################################
#### This script cleans the data required for feature creation ####
###################################################################

import pandas as pd

def get_cleaned_odds(df=None):
    # If a df hasn't been specified as a parameter, read the odds df
    if df is None:
        df = pd.read_csv("data/afl_odds.csv")

    # Get a dictionary of team names we want to change and their new values
    team_name_mapping = {
    'Adelaide Crows': 'Adelaide',
    'Brisbane Lions': 'Brisbane',
    'Carlton Blues': 'Carlton',
    'Collingwood Magpies': 'Collingwood',
    'Essendon Bombers': 'Essendon',
    'Fremantle Dockers': 'Fremantle',
    'GWS Giants': 'GWS',
    'Geelong Cats': 'Geelong',
    'Gold Coast Suns': 'Gold Coast',
    'Greater Western Sydney': 'GWS',
    'Greater Western Sydney Giants': 'GWS',
    'Hawthorn Hawks': 'Hawthorn',
    'Melbourne Demons': 'Melbourne', 
    'North Melbourne Kangaroos': 'North Melbourne',
    'Port Adelaide Magpies': 'Port Adelaide',
    'Port Adelaide Power': 'Port Adelaide', 
    'P Adelaide': 'Port Adelaide',
    'Richmond Tigers': 'Richmond',
    'St Kilda Saints': 'St Kilda', 
    'Sydney Swans': 'Sydney',
    'West Coast Eagles': 'West Coast',
    'Wetsern Bulldogs': 'Western Bulldogs',
    'Western Bullbogs': 'Western Bulldogs'
    }

    # Add columns
    df = (df.assign(date=lambda df: pd.to_datetime(df.trunc), # Create a datetime column
                    home_team=lambda df: df.path.str.extract('(([\w\s]+) v ([\w\s]+))', expand=True)[1].str.strip(),
                    away_team=lambda df: df.path.str.extract('(([\w\s]+) v ([\w\s]+))', expand=True)[2].str.strip())
            .drop(columns=['path', 'trunc', 'event_name']) # Drop irrelevant columns
            .rename(columns={'selection_name': 'team'}) # Rename columns
            .replace(team_name_mapping)
            .sort_values(by='date')
            .reset_index(drop=True)
            .assign(home_game=lambda df: df.apply(lambda row: 1 if row.home_team == row.team else 0, axis='columns'))
            .drop(columns=['home_team', 'away_team']))
    return df


def get_cleaned_match_results(df=None):
    # If a df hasn't been specified as a parameter, read the match_results df
    if df is None:
        df = pd.read_csv("data/afl_match_results.csv")

    # Create column lists to loop through - these are the columns we want in home and away dfs
    home_columns = ['Game', 'Date', 'Round.Number', 'Home.Team', 'Home.Goals', 'Home.Behinds', 'Home.Points', 'Margin', 'Venue', 'Away.Team', 'Away.Goals', 'Away.Behinds', 'Away.Points']
    away_columns = ['Game', 'Date', 'Round.Number', 'Away.Team', 'Away.Goals', 'Away.Behinds', 'Away.Points', 'Margin', 'Venue', 'Home.Team', 'Home.Goals', 'Home.Behinds', 'Home.Points']
    
    mapping = ['game', 'date', 'round', 'team', 'goals', 'behinds', 'points', 'margin', 'venue', 'opponent', 'opponent_goals', 'opponent_behinds', 'opponent_points']
    
    team_name_mapping = {
    'Brisbane Lions': 'Brisbane',
    'Footscray': 'Western Bulldogs'
    }

    # Create a df with only home games
    df_home = (df[home_columns]
                .rename(columns={old_col: new_col for old_col, new_col in zip(home_columns, mapping)})
                .assign(home_game=1))

    # Create a df with only away games
    df_away = (df[away_columns]
                .rename(columns={old_col: new_col for old_col, new_col in zip(away_columns, mapping)})
                .assign(home_game=0,
                        margin=lambda df: df.margin * -1))

    # Append these dfs together
    new_df = (df_home.append(df_away)
                     .sort_values(by='game') # Sort by game ID
                     .reset_index(drop=True) # Reset index
                     .assign(date=lambda df: pd.to_datetime(df.date)) # Create a datetime column
                     .replace(team_name_mapping)) # Rename team names to be consistent with other dfs
    return new_df


def get_cleaned_aggregate_player_stats(df=None):
    # If a df hasn't been specified as a parameter, read the player_stats df
    if df is None:
        df = pd.read_csv("data/afl_player_stats.csv")

    agg_stats = (df.rename(columns={ # Rename columns to lowercase
                    'Season': 'season',
                    'Round': 'round',
                    'Team': 'team',
                    'Opposition': 'opponent',
                    'Date': 'date'
                    })
                   .groupby(by=['date', 'season', 'team', 'opponent'], as_index=False) # Groupby to aggregate the stats for each game
                   .sum()
                   .drop(columns=['DE', 'TOG', 'Match_id']) # Drop columns
                   .assign(date=lambda df: pd.to_datetime(df.date, format="%d/%m/%Y")) # Create a datetime object
                   .sort_values(by='date')
                   .reset_index(drop=True))
    return agg_stats


def outlier_eliminator(df):
    # Eliminate Essendon 2016 games
    essendon_filter_criteria = ~(((df['team'] == 'Essendon') & (df['season'] == 2016)) | ((df['opponent'] == 'Essendon') & (df['season'] == 2016)))
    df = df[essendon_filter_criteria].reset_index(drop=True)
    
    return df


def prepare_afl_data():
    odds = get_cleaned_odds()
    match_results = get_cleaned_match_results()
    agg_stats = get_cleaned_aggregate_player_stats()

    merged_df = (odds[odds.team.isin(agg_stats.team.unique())]
                .pipe(pd.merge, match_results, on=['date', 'team', 'home_game'])
                .pipe(pd.merge, agg_stats, on=['date', 'team', 'opponent'])
                .sort_values(by=['game']))

    afl_data = outlier_eliminator(merged_df)

    # Tag each column that will be used in feature creation with 'f_' so we can easily grab these columns
    non_feature_cols = ['team', 'date', 'home_game', 'game', 'round', 'venue', 'opponent', 'season']
    afl_data = afl_data.rename(columns={col: 'f_' + col for col in afl_data if col not in non_feature_cols})

    return afl_data