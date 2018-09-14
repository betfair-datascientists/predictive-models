library(MASS)
library(ordinal)
library(fitzRoy)
library(tidyverse)

df<-fitzRoy::get_afltables_stats(start_date = "1897-01-01", end_date = Sys.Date())
df<-df%>%filter(Season>2010)

names(df)
team_stats<-df%>%
  dplyr::select(Date, First.name,Surname,Season, Round, Playing.for, Kicks:Goal.Assists)%>%
  group_by(Date, Season, Round, Playing.for)%>%
  summarise_if(is.numeric,funs(sum=c(sum(.))))

player_stats<-df%>%
  dplyr::select(Date, First.name,Surname,Season, Round, Playing.for, Kicks:Goal.Assists)

complete_df<-left_join(player_stats,team_stats, by=c("Date"="Date", "Season"="Season",  "Playing.for"="Playing.for"))

dataset_scores<-fitzRoy::match_results
names(dataset_scores)
dataset_scores1<-dataset_scores%>%dplyr::select (Date, Round, Home.Team, Home.Points,Game)
dataset_scores2<-dplyr::select(dataset_scores, Date, Round, Away.Team, Away.Points,Game)

colnames(dataset_scores1)[3]<-"Team"
colnames(dataset_scores1)[4]<-"Points"
colnames(dataset_scores2)[3]<-"Team"
colnames(dataset_scores2)[4]<-"Points"

df5<-rbind(dataset_scores1,dataset_scores2)
dataset_margins<-df5%>%group_by(Game)%>%
  arrange(Game)%>%
  mutate(margin=c(-diff(Points),diff(Points)))
# View(dataset_margins)
dataset_margins$Date<-as.Date(dataset_margins$Date)
complete_df$Date<-as.Date(complete_df$Date)

complete_df<-left_join(complete_df,dataset_margins,by=c("Date"="Date",  "Playing.for"="Team"))


complete_df_ratio<-complete_df%>%
  mutate(kick.ratio=Kicks/Kicks_sum,
         Marks.ratio=Marks/Marks_sum,
         handball.ratio=Handballs/Handballs_sum,
         Goals.ratio=Goals/Goals_sum,
         behinds.ratio=Behinds/Behinds_sum,
         hitouts.ratio=Hit.Outs/Hit.Outs_sum,
         tackles.ratio=Tackles/Tackles_sum,
         rebounds.ratio=Rebounds/Rebounds_sum,
         inside50s.ratio=Inside.50s/Inside.50s_sum,
         clearances.ratio=Clearances/Clearances_sum,
         clangers.ratio=Clangers/Clangers_sum,
         freefors.ratio=Frees.For/Frees.For_sum,
         freesagainst.ratio=Frees.Against/Frees.Against_sum,
         Contested.Possessions.ratio=Contested.Possessions/Contested.Possessions_sum,
         Uncontested.Possessions.ratio=Uncontested.Possessions/Uncontested.Possessions_sum,
         contested.marks.ratio=Contested.Marks/Contested.Marks_sum,
         marksinside50.ratio=Marks.Inside.50/Marks.Inside.50_sum,
         one.percenters.ratio=One.Percenters/One.Percenters_sum,
         bounces.ratio=Bounces/Bounces_sum,
         goal.assists.ratio=Goal.Assists/Goal.Assists_sum,
         disposals.ratio=(Kicks+Handballs)/(Kicks_sum+Handballs_sum))
df<-complete_df_ratio%>%dplyr::select(Date,Game, First.name, Surname, Season, Round.x, Playing.for,-Brownlow.Votes, Brownlow.Votes_sum,everything())
df<-df%>%dplyr::select(-Brownlow.Votes,everything())
df[is.na(df)] <- 0


in.sample  <- subset(df, Season %in% c(2013:2016))
# out.sample <- subset(df, Season == 2014)
in.sample$Brownlow.Votes <- factor(in.sample$Brownlow.Votes)

in.sample<-in.sample%>%filter(Round.x %in% c("1","2","3","4","5","6","7","8",
                                             "9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"))


names(in.sample)

in.sample$Player<-paste(in.sample$First.name,in.sample$Surname)

in.sample<-in.sample%>%dplyr::select(Player, Date, Season, Round.x, Playing.for, margin:Brownlow.Votes)
# in.sample<-in.sample[-c(1,2)]

temp1<-scale(in.sample[,8:29])
in.sample[,8:29]<-temp1
#attributes(temp1)
temp1.center<-attr(temp1,"scaled:center")
temp1.scale<-attr(temp1,"scaled:scale")






summary(in.sample)

fm1<-clm(Brownlow.Votes~ handball.ratio +  Marks.ratio +  
           disposals.ratio+  hitouts.ratio+
           freefors.ratio +  freesagainst.ratio +  tackles.ratio +	Goals.ratio +	behinds.ratio +	Contested.Possessions.ratio+
           Uncontested.Possessions.ratio +	clangers.ratio +	contested.marks.ratio +	marksinside50.ratio +
           clearances.ratio +	rebounds.ratio +	inside50s.ratio +	one.percenters.ratio +	bounces.ratio+
           goal.assists.ratio  +margin, 
         data = in.sample)

fm2<- stepAIC(fm1, direction='backward',type=AIC)


names(fitzRoy::player_stats)
df_2017<-fitzRoy::player_stats%>%
  filter(Season==2017)

team_stats_out<-df_2017%>%
  dplyr::select(Date, Player,Season, Round, Team, CP:T5)%>%
  group_by(Date,Season, Round, Team)%>%
  summarise_if(is.numeric,funs(sum=c(sum(.))))

player_stats_out<-df_2017%>%
  dplyr::select(Date, Player,Season, Round, Team, CP:T5)


complete_df_out<-left_join(player_stats_out,team_stats_out, by=c("Date"="Date", "Season"="Season",  "Team"="Team"))



dataset_scores<-fitzRoy::match_results
names(dataset_scores)
dataset_scores1<-dataset_scores%>%dplyr::select (Date, Round, Home.Team, Home.Points,Game)
dataset_scores2<-dplyr::select(dataset_scores, Date, Round, Away.Team, Away.Points,Game)

colnames(dataset_scores1)[3]<-"Team"
colnames(dataset_scores1)[4]<-"Points"
colnames(dataset_scores2)[3]<-"Team"
colnames(dataset_scores2)[4]<-"Points"


df5<-rbind(dataset_scores1,dataset_scores2)
dataset_margins<-df5%>%group_by(Game)%>%
  arrange(Game)%>%
  mutate(margin=c(-diff(Points),diff(Points)))
dataset_margins$Date<-as.Date(dataset_margins$Date)
complete_df_out$Date<-as.Date(complete_df_out$Date)

dataset_margins<-dataset_margins %>%mutate(Team = str_replace(Team, "Brisbane Lions", "Brisbane"))

dataset_margins<-dataset_margins %>%mutate(Team = str_replace(Team, "Footscray", "Western Bulldogs"))


complete_df_out<-left_join(complete_df_out,dataset_margins,by=c("Date"="Date",  "Team"="Team"))

names(complete_df_out)

####create the new ratios
complete_df_ratio_out<-complete_df_out%>%
  mutate(kick.ratio=K/K_sum,
         Marks.ratio=M/M_sum,
         handball.ratio=HB/HB_sum,
         Goals.ratio=G/G_sum,
         behinds.ratio=B/B_sum,
         hitouts.ratio=HO/HO_sum,
         tackles.ratio=T/T_sum,
         rebounds.ratio=R50/R50_sum,
         inside50s.ratio=I50/I50_sum,
         clearances.ratio=(CCL+SCL)/(CCL_sum+SCL_sum),
         clangers.ratio=CL/CL_sum,
         freefors.ratio=FF/FF_sum,
         freesagainst.ratio=FA/FA_sum,
         Contested.Possessions.ratio=CP/CP_sum,
         Uncontested.Possessions.ratio=UP/UP_sum,
         contested.marks.ratio=CM/CM_sum,
         marksinside50.ratio=MI5/MI5_sum,
         one.percenters.ratio=One.Percenters/One.Percenters_sum,
         bounces.ratio=BO/BO_sum,
         goal.assists.ratio=GA/GA_sum,
         disposals.ratio=D/D_sum)




conforming<-complete_df_ratio_out%>%
  dplyr::select(Player, Date, Season, Round.x, Team, margin, 
                kick.ratio:disposals.ratio)

conforming$Brownlow.Votes<-0
out.sample=conforming

newdata   <- out.sample[ , -ncol(out.sample)]

newdata[,6:27]<-scale(newdata[,6:27],center=temp1.center,scale=temp1.scale) 

pre.dict    <- predict(fm2,newdata=newdata, type='prob')
pre.dict.m  <- data.frame(matrix(unlist(pre.dict), nrow= nrow(newdata)))
colnames(pre.dict.m) <- c("vote.0", "vote.1", "vote.2", "vote.3")

newdata.pred  <- cbind.data.frame(newdata, pre.dict.m)



#### Step 1: Get expected value on Votes
newdata.pred$expected.votes <- newdata.pred$vote.1 + 2*newdata.pred$vote.2 + 3*newdata.pred$vote.3

####Join back on matchID whoops!

get_match_ID<-fitzRoy::player_stats

xx<-get_match_ID%>%dplyr::select(Date, Player, Match_id)
newdata.pred<-left_join(newdata.pred, xx, by=c("Date"="Date",  "Player"="Player"))



newdata.pred<-filter(newdata.pred, Date<"2017-09-01")


sum1 <- aggregate(vote.1~Match_id, data = newdata.pred, FUN = sum ); names(sum1) <- c("Match_id", "sum.vote.1");
sum2 <- aggregate(vote.2~Match_id, data = newdata.pred, FUN = sum ); names(sum2) <- c("Match_id", "sum.vote.2");
sum3 <- aggregate(vote.3~Match_id, data = newdata.pred, FUN = sum ); names(sum3) <- c("Match_id", "sum.vote.3");

#### Step 3: Add sum of each vote by matchId to big table
newdata.pred <- merge(newdata.pred, sum1, by = "Match_id")
newdata.pred <- merge(newdata.pred, sum2, by = "Match_id")
newdata.pred <- merge(newdata.pred, sum3, by = "Match_id")

#### Step 4: Add std1/2/3
newdata.pred$std.1  <- (newdata.pred$sum.vote.1/newdata.pred$vote.1)^-1
newdata.pred$std.2  <- (newdata.pred$sum.vote.2/newdata.pred$vote.2)^-1
newdata.pred$std.3  <- (newdata.pred$sum.vote.3/newdata.pred$vote.3)^-1


#### Step 5: Expected standard game vote
newdata.pred$exp_std_game_vote <- newdata.pred$std.1 + 2*newdata.pred$std.2 + 3*newdata.pred$std.3  


#### Step 6: List of winners

newdata.pred$PlayerName<-paste(newdata.pred$Player," ",newdata.pred$Team)
winners.stdgame   <- aggregate(exp_std_game_vote~PlayerName, data = newdata.pred, FUN = sum );
winners.stdgame   <- winners.stdgame[order(-winners.stdgame$exp_std_game_vote), ]
winners.stdgame[1:10, ]
