###################################################
### World Cup 2018 Machine Learning Model
###
### This script trains a simple machine learning model on data
### before the 2014 world cup and tests it's predictions
### on that competition
###################################################

library(readr)
library(dplyr)
library(tidyr)
library(lubridate)
library(snakecase)
library(MLmetrics)
library(RcppRoll)
library(randomForest)

# Read in the world cup CSV data from out github page
rawdata = read_csv("https://raw.githubusercontent.com/betfair-datascientists/Predictive-Models/master/World-Cup-2018/data/wc_datathon_dataset.csv?_sm_au_=iVV7ZLQVWLNsk7WQ")

# Convert the match date to an R date
# -- and add in a unique match key for later
rawdata = rawdata %>% 
  mutate(
    date = dmy(rawdata$date),
    match_key = to_snake_case(paste(format(date, "%Y %m %d"), team_1, team_2))
  )

# +++++++++++++++++
# Restructuring Data
# +++++++++++++++++

# ML solutions are a bit awkward for sport problems because the data is often on a single row
# To simplify our calculations we'll split out the data into 2 rows before collapsing it later for predictions

team_1_df = rawdata %>%
  mutate(side = "team_1") %>%
  select(
    match_key, side, tournament, date,
    "team" = team_1, "opponent" = team_2, 
    "goals" = team_1_goals, "opp_goals" = team_2_goals, 
    "home_game" = is_team_1_home, is_neutral
  )

team_2_df = rawdata %>%
  mutate(side = "team_2") %>%
  select(
    match_key, side, tournament, date,
    "team" = team_2, "opponent" = team_1, 
    "goals" = team_2_goals, "opp_goals" = team_1_goals, 
    "home_game" = is_team_2_home, is_neutral
  )

expanded_df = team_1_df %>% union_all(team_2_df)
head(expanded_df)

# +++++++++++++++++
# Calculate Features
# +++++++++++++++++

# We'll use this expanded dataframe to perform the feature calculations
# including looking back at previous games and performing rolling averages for example


expanded_df = expanded_df %>%
  mutate(
    margin = goals - opp_goals,
    home_game = ifelse(home_game, 1, 0),
    is_neutral = ifelse(is_neutral, 1, 0),
    target = as.factor(
      case_when(
        margin > 0 ~ "Win",
        margin < 0 ~ "Lose",
        TRUE ~ "Draw"
      )
    )
  ) %>%
  group_by(team) %>%
  mutate(
    last_margin = lag(margin, 1, default = 0),
    last_5_margin_mean = roll_mean(margin, 5, align = "right", fill = NA),
    max_score_10 = roll_max(goals, 10, align = "right", fill = NA),
    max_allowed_10 = roll_max(opp_goals, 10, align = "right", fill = NA)
  ) %>%
  ungroup() %>%
  drop_na() 

# +++++++++++++++++++++++++++++++++++++
# Assemble features together
# +++++++++++++++++++++++++++++++++++++

# Now that we've performed the calculations we'll flatten out the data again
# by joining the team 1 and team 2 features back together into a single row

# Outcome
outcome_df = expanded_df %>%
  filter(side == "team_1") %>%
  select(match_key, date, tournament, margin, target, home_game, is_neutral)

# Need team 1 feature matrix
team_1_features = expanded_df %>%
  filter(side == "team_1") %>%
  select(match_key, last_margin:max_allowed_10) %>%
  rename_at(vars(-match_key), function(x){paste0("h_", x)})

# Need team 2 features
team_2_features = expanded_df %>%
  filter(side == "team_2") %>%
  select(match_key, last_margin:max_allowed_10) %>%
  rename_at(vars(-match_key), function(x){paste0("a_", x)})

# Join together into feature matrix
feature_matrix = outcome_df %>%
  inner_join(team_1_features)%>%
  inner_join(team_2_features)

# ++++++++++++++++++++
# Split Training / Test
# ++++++++++++++++++++

# We're going to use all matches (qualifiers / friendlies etc) AFTER the 2010 world cup to predict the 2014 world cup matches
training = feature_matrix %>%
    filter(
      # Match should occur after 2010 world cup
      date > rawdata %>% filter(tournament == "World Cup 2010") %>% pull(date) %>% max(), 
      # Match should occur before 2014 world cup starts
      date < rawdata %>% filter(tournament == "World Cup 2014") %>% pull(date) %>% min()
     ) %>%
     # Select only features and target
     select(
       target,
       is_neutral, home_game,
       # Select feature columns that start with an a or h
       matches("^h|^a")
     )

# We'll be testing our appraoch on the 2014 World Cup
wc_2014 = feature_matrix %>% filter(tournament == "World Cup 2014")

# ++++++++++++++++
#   Train Model
# ++++++++++++++++

# For reproducibility
set.seed(123)

# Train Model
rf.fit = randomForest(target ~ . , data = training)

# Create predictions for test set / wc 2015
rf.pred = predict(rf.fit, wc_2014, type = "prob")

# ++++++++++++
# Assess Model
# ++++++++++++

final = wc_2014 %>%
  # Binary yes / no for each of the home loss results
  mutate(
    home_actual = ifelse(margin > 0, 1, 0),
    draw_actual = ifelse(margin == 0, 1, 0),
    away_actual = ifelse(margin < 0, 1, 0)
  ) %>%
  # Predicted Probabilities
  mutate(
    home_prob = rf.pred[,3],
    away_prob = rf.pred[,2],
    draw_prob = rf.pred[,1] 
  ) 

# Calculate the log-loss for this tournament
MultiLogLoss(
  y_pred = final[,c("home_prob", "draw_prob", "away_prob")] %>% as.matrix(),
  y_true = final[,c("home_actual", "draw_actual", "away_actual")] %>% as.matrix()
)

# Achieved a log loss of: 0.808671
# This would be very competitive!


