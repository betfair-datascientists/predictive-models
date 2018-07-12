

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
    Game        home_team         away_team Predicted Winner  home_odds  \
0  15334         St Kilda           Carlton         St Kilda       1.40   
1  15335         Hawthorn          Brisbane         Hawthorn       1.27   
2  15336        Melbourne  Western Bulldogs        Melbourne       1.19   
3  15337       Gold Coast          Essendon         Essendon       5.60   
4  15338              GWS          Richmond         Richmond       2.84   
5  15339      Collingwood        West Coast      Collingwood       1.44   
6  15340  North Melbourne            Sydney           Sydney       2.22   
7  15341        Fremantle     Port Adelaide    Port Adelaide       4.60   

   away_odds     home_elo     away_elo  
0       3.50  1422.699826  1249.516972  
1       4.80  1601.212897  1264.807130  
2       8.20  1489.199372  1463.143986  
3       1.23  1293.271340  1496.279191  
4       1.55  1583.093084  1623.317097  
5       2.68  1508.140954  1592.391714  
6       1.82  1472.869065  1662.752549  
7       1.28  1426.443873  1583.704866  
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
