
# EPL Predictions Model
These tutorials will walk you through how to create your own EPL predictive model, as well as an automated betting strategy. The output will be the modelled odds for each EPL match, which will be posted on the [Hub](https://www.betfair.com.au/hub/tools/models/epl-predictions-model/).

For the interactive tutorials, download and walk through the Jupyter notebook files (.ipynb). If you want to jump ahead, we have also created scripts which prepare all the data from the previous tutorial. These are the (.py) files.

The goal for future iterations of this tutorial is to create other models too, to allow betting on other markets like Total Goals Over/Under, FT/HT markets etc.

For now, however, these tutorials will allow you to create your own predictions for the 2018/19 EPL season.

## Requirements
There are a few requirements to run the predictive scripts and walk through the interactive tutorials on your local computer. 

If you are happy to read through the tutorials on Github, but not run the code yourself, you simply need to click the tutorial links and the tutorial can be viewed in your browser. However, if you are keen to be able to run the code yourself and try different things out, you will need to install the following:
* Python 
* Jupyter Notebook (Installed through the Anaconda Distribution)

If you don't already have Python installed, we advise you to install it through [Anaconda](https://www.anaconda.com/download/). This also installs Jupyter and is super convenient.

## 01. Data Acquisition & Exploration
This tutorial (yet to be published) will guide you through how to pull data from public resources online, as well as the exploratory data analysis phase of the Data Science process. We will look at how certain teams have performed historically, and gain an in depth understanding of our dataset.

## 02. Data Preparation & Feature Engineering
[This tutorial](insert_link_here) will walk you through the data wrangling and feature engineering process. We will create exponentially weighted averages and optimise the weighting based on log loss. Other features discussed in other tutorials will also be created, such as a margin weighted Elo ranking and Elo weighted goals for/against.

## 03. Model Building & Hyperparameter Tuning
[This tutorial](insert_link_here) will take our cleaned data/features created previously and try out different algorithms to see how effectively they model EPL matches.

## 04. Weekly Predictions
[This tutorial](insert_link_here) will create weekly predictions for each Gameweek. These will be posted on [The Hub](https://www.betfair.com.au/hub/tools/models/epl-predictions-model/). You can also view what the model predicted in previous Gameweeks.

## 05. Analysing Predictions & Staking Strategies
[This tutorial](insert_link_here) will analyse our predictions for the previous season and focus on iterative improvement. We will also analyse if the model is profitable and try to create profitable betting strategies which we can implement with Betfair's API.