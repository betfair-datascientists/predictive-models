###################################################################
#### This script cleans the data required for feature creation ####
###################################################################

# Import libraries
import pandas as pd
'''
We will first clean each DataFrame to ensure that the team names are consistent across DataFrames. 
Note that the player_stats DataFrame does not need cleaning as it has standard team names
'''

##########################################################################################
#### Create functions which clean the odds, match_results and player_stats DataFrames ####
##########################################################################################

def clean_match_results(match_results_path):
    df = pd.read_csv(match_results_path)
    
    # Clean team names to be consistent across DataFrames
    df = df.replace(
    {
        'Brisbane Lions': 'Brisbane',
        'Footscray': 'Western Bulldogs'
    }
    )
    return df

def odds_wrangling(df):
    # Create a date column
    df['Date'] = pd.to_datetime(df['trunc']).dt.date
    
    # Grab the home and away teams using regex from the match_results column
    df['home_team'] = df['path'].str.extract('(([\w\s]+) v ([\w\s]+))', expand=True)[1].str.strip()
    df['away_team'] = df['path'].str.extract('(([\w\s]+) v ([\w\s]+))', expand=True)[2].str.strip()
    df['match_details'] = df['path'].str.extract('(([\w\s]+) v ([\w\s]+))', expand=True)[0].str.strip()
    
    # Drop unneeded columns
    df = df.drop(columns=['path', 'trunc', 'event_name', 'match_details'])
    
    # Rename column
    df = df.rename(columns={'selection_name': 'Team'})
    return df

def clean_odds(df):
    # Clean team names to be consistent across DataFrames
    df = df.replace(
    {
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
    )
    return df

def match_results_wrangling(df):
    # Create DataFrame which includes all the home teams' statistics, as well as the stats for the away team (Opposition)
    df_home = pd.DataFrame(
        {
            'Game': df['Game'],
            'Date': df['Date'],
            'Round': df['Round.Number'],
            'Team': df['Home.Team'],
            'Goals': df['Home.Goals'],
            'Behinds': df['Home.Behinds'],
            'Points': df['Home.Points'],
            'Margin': df['Margin'],
            'Venue': df['Venue'],
            'Home?': 1,
            'Opposition': df['Away.Team'],
            'Opposition Goals': df['Away.Goals'],
            'Opposition Behinds': df['Away.Behinds'],
            'Opposition Points': df['Away.Points']
    })
    # Create DataFrame which includes all the away teams' statistics, as well as the stats for the home team (Opposition)
    df_away = pd.DataFrame(
        {
            'Game': df['Game'],
            'Date': df['Date'],
            'Round': df['Round.Number'],
            'Team': df['Away.Team'],
            'Goals': df['Away.Goals'],
            'Behinds': df['Away.Behinds'],
            'Points': df['Away.Points'],
            'Margin': - df['Margin'],
            'Venue': df['Venue'],
            'Home?': 0,
            'Opposition': df['Home.Team'],
            'Opposition Goals': df['Home.Goals'],
            'Opposition Behinds': df['Home.Behinds'],
            'Opposition Points': df['Home.Points']
    })
    
    # Append the DataFrames together, then sort by the Game ID so that we have the same game on consecutive rows
    df = df_home.append(df_away).sort_values(by='Game').reset_index(drop=True)
    
    # Change the Date column to a Datetime object
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df

# Define a function which cleans the match_results DataFrame
def clean_match_results(df):
    # Clean team names to be consistent across DataFrames
    df = df.replace(
    {
        'Brisbane Lions': 'Brisbane',
        'Footscray': 'Western Bulldogs'
    }
    )  
    return df

############################################################################################
#### Create functions which wrangle each DataFrame with the goal to merge them into one ####
############################################################################################

# Define a function which gets data from each of the player_stats data sources; one from 2010-2017 and one from 2018-onwards
def combine_player_stats_dfs(old_player_stats_path, new_player_stats_path):
	# Read the historical player stats DataFrames together and append them
	df = pd.read_csv(old_player_stats_path)
	df_new = pd.read_csv(new_player_stats_path)
	col_order = df.columns
	df = df.append(df_new)[col_order]
	return df

# Define a function which aggregates the player_stats into individual game stats
def player_stats_wrangling(df):
    # Aggregate the stats
    agg_stats = df.groupby(by=['Date', 'Season', 'Round', 'Team', 'Opposition', 'Status'], as_index=False).sum()

    # Drop irrelevant columns such as Disposal Efficiency and Time On Ground which are meaningless when aggregated
    agg_stats = agg_stats.drop(columns=['DE', 'TOG', 'Match_id'])
	
	# Change the Date column to a Datetime object
    agg_stats['Date'] = pd.to_datetime(agg_stats['Date']).dt.date
    return agg_stats
	
#####################################################################################
#### Create functions which use the previous functions to prepare each DataFrame ####
#####################################################################################

def prepare_odds(odds_path):
	df = pd.read_csv(odds_path)
	df = clean_odds(df)
	df = odds_wrangling(df)
	return df
	
def prepare_match_results(match_results_path):
	df = pd.read_csv(match_results_path)
	df = clean_match_results(df)
	df = match_results_wrangling(df)
	return df

def prepare_agg_stats(old_player_stats_path, new_player_stats_path):
	df = combine_player_stats_dfs(old_player_stats_path, new_player_stats_path)
	df = player_stats_wrangling(df)
	return df

# Create a function which merges the DataFrames
def merge_dfs(odds_df, match_results_df, agg_stats_df):
    # Before we merge the DataFrames, let's filter out games that aren't played between teams in our agg_stats_df
    teams = agg_stats_df['Team'].unique()
    odds_df = odds_df[(odds_df['home_team'].isin(teams)) & (odds_df['away_team'].isin(teams))]
    
    # Merge the odds DataFrame with match_results
    df = pd.merge(odds_df, match_results_df, how='inner', on=['Team', 'Date'])
    
    # Merge that df with agg_stats
    df = pd.merge(df, agg_stats_df, how='inner', on=['Team', 'Date'])
    
    # Sort the values so that each game is ordered by Date
    df = df.sort_values(by=['Game', 'Home?']).reset_index(drop=True)
    
    # Drop duplicate columns and rename these
    df = df.drop(columns=['Round_y', 'Opposition_y']).rename(columns={'Opposition_x': 'Opposition', 'Round_x': 'Round'})
    return df

# Define a function which eliminates outliers such as Essendon's 2016 season where their team was banned for a year
def outlier_eliminator(df):
    # Eliminate Essendon 2016 games
    essendon_filter_criteria = ~(((df['Team'] == 'Essendon') & (df['Season'] == 2016)) | ((df['Opposition'] == 'Essendon') & (df['Season'] == 2016)))
    df = df[essendon_filter_criteria]
    
    # Reset index
    df = df.reset_index(drop=True)
    return df

# Define our final function; this uses all the other functions we defined to prepare a final DataFrame with beautifully clean data, ready for analysis/feature creation
def prepare_afl_data():
	odds = prepare_odds("data/afl_odds.csv")
	match_results = prepare_match_results("data/afl_match_results.csv")
	agg_stats = prepare_agg_stats("data/player_stats_2010.csv", "data/player_stats_2018.csv")
	afl_data = merge_dfs(odds, match_results, agg_stats)
	afl_data = outlier_eliminator(afl_data)
	return afl_data

