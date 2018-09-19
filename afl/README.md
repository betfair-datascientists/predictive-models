

# AFL Weekly Predictive Model
These tutorials will walk you through how to create your own basic AFL predictive model. We will use Python as our language of choice. The output will be modelled odds for each game, which will be posted below weekly.

For the interactive tutorials, download and walk through the Jupyter notebook files (.ipynb). If you want to jump ahead, we have also created scripts which prepare all the data from the previous tutorial. These are the (.py) files.

The goal for future iterations of this tutorial is to create automated betting strategy on Betfair. Stay tuned for updates.

## This Week's Predictions
The predictions below are created from the afl_modelling script. For a detailed walkthrough of how we got these predictions, please have a look at the tutorials below.
```Python
# This week's predictions
preds = afl_modelling.prepare_afl_predictions_df()
print(preds)
'''
        date   home_team    away_team          venue  modelled_home_odds  \
0 2018-09-21    Richmond  Collingwood            MCG            1.304895   
1 2018-09-22  West Coast    Melbourne  Optus Stadium            1.895771   

   modelled_away_odds  odds  odds_away   f_elo_home   f_elo_away  
0            4.279819  1.42       3.40  1707.546289  1565.277390  
1            2.116357  1.88       2.12  1638.076814  1576.417977  
'''
```

## Requirements
There are a few requirements to run the predictive scripts and walk through the interactive tutorials on your local computer. 

If you are happy to read through the tutorials on Github, but not run the code yourself, you simply need to click the tutorial links and the tutorial can be viewed in your browser. However, if you are keen to be able to run the code yourself and try different things out, you will need to install the following:
* Python 
* Jupyter Notebook (Installed through the Anaconda Distribution)

If you don't already have Python installed, we advise you to install it through [Anaconda](https://www.anaconda.com/download/). This also installs Jupyter and is super convenient.

## AFL Data Cleaning Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/01.%20afl_data_cleaning_tutorial.ipynb) will walk you through the first steps of a typical data science project - exploring and cleaning your data. We will grab our Aussie Rules football data which include player statistics, match results (obtained through the [FitzRoy](https://github.com/jimmyday12/fitzRoy) package) and game odds to clean it so that it is ready for feature creation and modelling.

If you are interested, feel free to explore the datasets on your own. The match results data dates back all the way to 1897, whilst the player statistics and odds data dates back to 2011.

If you want to skip ahead to the feature creation tutorial, simply download the afl_data_cleaning script and move onto the next tutorial. This script will clean all the data for you in the next tutorial.

Note: The columns used in these tutorials refer to the following statistics:

```Python
# Column abbreviations
column_abbreviations = pd.read_csv("data/afl_data_columns_mapping.csv")
print(column_abbreviations)
'''
   Feature Abbreviated                  Feature
0                   GA             Goal Assists
1                   CP    Contested Possessions
2                   UP  Uncontested Possessions
3                   ED      Effective Disposals
4                   CM          Contested Marks
5                  MI5          Marks Inside 50
6       One.Percenters           One Percenters
7                   BO                  Bounces
8                    K                    Kicks
9                   HB                Handballs
10                   D                Disposals
11                   M                    Marks
12                   G                    Goals
13                   B                  Behinds
14                   T                  Tackles
15                  HO                  Hitouts
16                 I50               Inside 50s
17                  CL               Clearances
18                  CG                 Clangers
19                 R50              Rebound 50s
20                  FF                Frees For
21                  FA            Frees Against
22                  AF       AFL Fantasy Points
23                  SC        Supercoach Points
24                 CCL        Centre Clearances
25                 SCL      Stoppage Clearances
26                  SI       Score Involvements
27                  MG            Metres Gained
28                  TO                Turnovers
29                 ITC               Intercepts
30                  T5        Tackles Inside 50
'''
```

## AFL Feature Creation Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/02.%20afl_feature_creation_tutorial.ipynb) will take our cleaned data which we created in the previous tutorial by using the afl_data_cleaning script and create features to be used in modelling.
It will be informative and walk you through the process up until the modelling stage.

If you want to skip ahead to the modelling tutorial, simply download the afl_data_cleaning and afl_feature_creation scripts and move onto the next tutorial. 

## AFL Modelling Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/03.%20afl_modelling.ipynb) walks you through the modelling phase where we try out a range of different algorithms for predictions. We then choose and optimise the best algorithms for our task, which we use to create predictions in the final tutorial.

## AFL Predictions Tutorial
Finally, [this tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/04.%20afl_weekly_predictions.ipynb) walks you through the process of generating weekly predictions/odds for AFL games. In future iterations of this tutorial we will add a how-to on setting up an automated betting strategy using your generated predictions and Betfair's API.
