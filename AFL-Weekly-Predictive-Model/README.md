

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
          home_team    away_team Predicted Winner     home_elo     away_elo  \
0            Sydney      Geelong           Sydney  1693.734358  1667.682944   
1          Richmond     Adelaide         Richmond  1609.677631  1578.060630   
2          Brisbane      Carlton         Brisbane  1233.037199  1280.164173   
3     Port Adelaide     St Kilda    Port Adelaide  1589.121018  1400.692014   
4  Western Bulldogs     Hawthorn         Hawthorn  1446.605223  1606.319440   
5         Melbourne    Fremantle        Melbourne  1494.602682  1449.483864   
6   North Melbourne   Gold Coast  North Melbourne  1472.379995  1285.335813   
7          Essendon  Collingwood      Collingwood  1478.524824  1528.013453   
8        West Coast          GWS       West Coast  1618.106496  1568.458243
'''
```

## Requirements
There are a few requirements to run the predictive scripts and walk through the interactive tutorials. Your best bet is to install these from the following links, if you haven't already.
* Python 
* Jupyter Notebook (Installed through the Anaconda Distribution)

If you don't already have Python installed, we advise you to install it through [Anaconda](https://www.anaconda.com/download/). This also install Jupyter and is super convenient.

## AFL Data Cleaning Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/01.%20afl_data_cleaning_tutorial.ipynb) will walk you through the first steps of a typical data science project - exploring and cleaning your data. We will grab our Aussie Rules football data which include player statistics, match results (obtained through the [FitzRoy](https://github.com/jimmyday12/fitzRoy) package) and game odds to clean it so that it is ready for feature creation and modelling.

If you are interested, feel free to explore the datasets on your own. The match results data dates back all the way to 1897, whilst the player statistics and odds data dates back to 2011.

If you want to skip ahead to the feature creation tutorial, simply download the afl_data_cleaning script and move onto the next tutorial. This script will clean all the data for you in the next tutorial.

## AFL Feature Creation Tutorial
[This tutorial](https://github.com/betfair-datascientists/Predictive-Models/blob/master/AFL-Weekly-Predictive-Model/02.%20afl_feature_creation_tutorial.ipynb) will take our cleaned data which we created in the previous tutorial by using the afl_data_cleaning script and create features to be used in modelling.
It will be informative and walk you through the process up until the modelling stage.

If you want to skip ahead to the modelling tutorial, simply download the afl_data_cleaning and afl_feature_creation scripts and move onto the next tutorial. 

## AFL Modelling Tutorial
[This tutorial](add_link) walks you through the modelling phase where we try out a range of different algorithms for predictions. We then choose and optimise the best algorithms for our task, which we use to create predictions in the final tutorial.

## AFL Predictions Tutorial
Finally, [this tutorial](add_link) walks you through the process of generating weekly predictions for AFL games. In future iterations of this tutorial we will add a how-to on setting up an automated betting strategy using your predictions generated and Betfair's API. These predictions will be posted on The Hub weekly, as well as on this page.
