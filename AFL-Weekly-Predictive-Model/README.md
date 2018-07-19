

# AFL Weekly Predictive Model
These tutorials will walk you through how to create your own basic AFL predictive model. We will use Python as our language of choice. The output will be odds for each team to win and the predicted winner, which will be shown on [The Hub](https://www.betfair.com.au/hub/tools/models/afl-prediction-model/) and posted below weekly.

For the interactive tutorials, download and walk through the Jupyter notebook files (.ipynb). If you want to jump ahead, we have also created scripts which prepare all the data from the previous tutorial. These are the (.py) files.

The goal for future iterations of this tutorial is to create an automated betting strategy on Betfair using our predictions which we generated using Machine Learning and Betfair's API. Currently, the tutorial can predict the winner of future AFL games. Stay tuned for updates.

## This Week's Predictions
The predictions below are created from the afl_modelling script. For a detailed walkthrough of how we got these predictions, please have a look at the tutorials below.
```Python
# This week's predictions
preds = afl_modelling.prepare_afl_predictions_df()
preds
'''
       home_team         away_team Predicted Winner  home_odds  away_odds  \
0       St Kilda          Richmond         Richmond       8.00       1.14   
1    Collingwood   North Melbourne      Collingwood       1.59       2.64   
2         Sydney        Gold Coast           Sydney       1.06      18.00   
3       Essendon         Fremantle         Essendon       1.24       5.30   
4       Brisbane          Adelaide         Brisbane       2.36       1.75   
5        Geelong         Melbourne        Melbourne       1.63       2.58   
6        Carlton          Hawthorn         Hawthorn       7.00       1.17   
7     West Coast  Western Bulldogs       West Coast       1.13       9.40   
8  Port Adelaide               GWS    Port Adelaide       1.90       2.10   

      home_elo     away_elo  
0  1429.168961  1609.933983  
1  1498.995146  1466.844032  
2  1668.777582  1287.580776  
3  1501.969755  1443.532609  
4  1285.782360  1633.965757  
5  1633.190328  1500.301134  
6  1243.047837  1580.237667  
7  1601.537522  1452.042224  
8  1566.616129  1596.476198  
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

## AFL Feature Creation Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/02.%20afl_feature_creation_tutorial.ipynb) will take our cleaned data which we created in the previous tutorial by using the afl_data_cleaning script and create features to be used in modelling.
It will be informative and walk you through the process up until the modelling stage.

If you want to skip ahead to the modelling tutorial, simply download the afl_data_cleaning and afl_feature_creation scripts and move onto the next tutorial. 

## AFL Modelling Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/03.%20afl_modelling.ipynb) walks you through the modelling phase where we try out a range of different algorithms for predictions. We then choose and optimise the best algorithms for our task, which we use to create predictions in the final tutorial.

## AFL Predictions Tutorial
Finally, [this tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/04.%20afl_weekly_predictions.ipynb) walks you through the process of generating weekly predictions for AFL games. In future iterations of this tutorial we will add a how-to on setting up an automated betting strategy using your generated predictions and Betfair's API. These predictions will be posted on [The Hub](https://www.betfair.com.au/hub/tools/models/afl-prediction-model/) weekly, as well as on this page.
