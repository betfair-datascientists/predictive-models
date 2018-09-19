# Brownlow

## Analysis Of AFL's Modelling Walkthrough
These tutorials will walk you through how to model the Brownlow Medal in R. We have teamed up with [Analysis Of AFL](https://twitter.com/anoafl), the co-creator of the [fitzRoy](https://github.com/jimmyday12/fitzRoy) package, to create an end-to-end walk through of how he modelled the Brownlow.

You are welcome to grab this code and chop and change the variables to how you see fit. Feel free to add additional variables and try to improve the model to make it your own.

The [takinghomecharlie](https://github.com/betfair-datascientists/predictive-models/blob/master/brownlow/takinghomecharlie_end_to_end.R) R script walks you through the modelling process. It is reproducible on any computer and uses the fitzRoy package to grab data.

The [takinghomecharlie R Markdown Script](https://github.com/betfair-datascientists/predictive-models/blob/master/brownlow/takinghomecharlie.Rmd) has detailed explanations of each step - although you will need to render this yourself in R Studio.

Alternatively, we have rendered it for you, simply download the [Taking Home Charlie HTML Document](https://github.com/betfair-datascientists/predictive-models/blob/master/brownlow/Taking%20Home%20Charlie.html) and read along on any browser. Note that unlike Jupyter Notebooks, this won't render on GitHub.

## Betfair Data Scientist's Modelling Walkthrough
[This tutorial](https://github.com/betfair-datascientists/predictive-models/blob/master/brownlow/Betfair%20Data%20Scientists'%20Brownlow%20Model.ipynb) will walk you through the EDA, feature creation and modelling process, and allow you to generate your own predictions for any year between 2012 and 2018. Below are the predicted top 15 for this year.

```Python

print(agg_predictions_2018.head(15))
           player              team  predicted_votes_scaled  match_id
0      T Mitchell          Hawthorn               35.484614        20
1          M Gawn         Melbourne               21.544278        22
2        D Martin          Richmond               20.444488        19
3        B Grundy       Collingwood               19.543511        22
4        C Oliver         Melbourne               19.009628        20
5        J Macrae  Western Bulldogs               18.931594        17
6   P Dangerfield           Geelong               18.621242        21
7         D Beams          Brisbane               17.621222        15
8           E Yeo        West Coast               16.015638        20
9         L Neale         Fremantle               15.495083        21
10         A Gaff        West Coast               15.165629        18
11      D Heppell          Essendon               15.083797        19
12      J Selwood           Geelong               14.989096        18
13   S Sidebottom       Collingwood               14.863136        18
14         N Fyfe         Fremantle               14.692243        11
```

### Requirements
Note that there are a few library requirements to run the python notebook. These are:
* pandas
* [h2o](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/downloading.html)
* numpy
* sklearn
* pickle

Simply use pip install to install these libraries, or google how to install them. For h2o specifically, click the link and follow the instructions.
