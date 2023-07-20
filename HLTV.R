################################################################################################################################################################################################
################################################################################################################################################################################################
####################################################################################  LOADING PACKAGES #########################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################

library("ISLR")
library("locfit")
library("systemfit")
library("nlme")
library("strucchange")
library("tseries")
library("lmtest")
library("urca")
library("sandwich")
library("stats")
library("aTSA")
library("FinTS")
library("bstats")
library("plm")
library("ts")
library("margins")
library("generalhoslem")
library("broom")
library("tidyverse")
library("car")
library("effects")
library("ggplot2")
library("dplyr")
library("AER")
library("stargazer")
library("DescTools")
library("aod")
library("ltm")
library("glmnet")
library("rugarch")
library("forecast")
library("fGarch")
library("scales")
library("selectiveInference")
library("ltm")
library("crch")
library("httr")
library("jsonlite")
library("purrr")
library("tidyr")
library("xlsx")
library("repurrrsive")
library("stringr")
library("boot")
library("censReg")
library("sampleSelection")
library("quantreg")
library("survival")
library("lawstat")
library("tweedie")
library("glmmTMB")
library("SGL")
library("cowplot")

################################################################ DATA #######################################################################

data <- read.csv("C:/Users/Honzík/OneDrive - Vysoká škola ekonomická v Praze/Connection/Plocha/Doktorát/HLTV/data_hltv_liq_2.csv", header = TRUE, stringsAsFactors = FALSE, sep = ",")
colnames(data)
data <- data[,c(1:33,40)]
colnames(data) <- c("Tournament", "N", "Tier", "Stage", "Type", "Date", "Year", "Team_1", "Team_2", "Team_1_rank", "Team_2_rank", "Rank_Difference", "Map_1" , "Team_1_score_map_1", "Team_2_score_map_1",
                    "Map_2", "Team_1_score_map_2", "Team_2_score_map_2", "Map_3", "Team_1_score_map_3", "Team_2_score_map_3", "Map_4", "Team_1_score_map_4", "Team_2_score_map_4", "Map_5",
                    "Team_1_score_map_5", "Team_2_score_map_5", "Format_BO", "Team_1_points", "Team_2_points", "Total_points", "Point_difference", "Prizepool", "Prizepool_Adjusted")

length(unique(data[,'Tournament']))
nrow(data[data['Stage'] == 'Finals',])

data <- data[,-c(1, 2, 6, 7, 8, 9, 13, 16, 19, 22, 25, 29:34)]
data <- data[!(is.na(data$Team_1_rank)),]
data <- data[!(is.na(data$Team_2_rank)),]

data$Stage <- factor(data$Stage, levels=c('Quarterfinals', 'Semifinals', 'Finals'))
data$Type <- factor(data$Type, levels=c('Online', 'Offline'))
data$Tier <- factor(data$Tier, levels=c('A', 'S'))
data$Format_BO <- factor(data$Format_BO, levels=c(3, 5))
head(data)

data$KLAASEN_1 <- 8 - log2(data$Team_1_rank)
data$KLAASEN_2 <- 8 - log2(data$Team_2_rank)
data$KLAASEN_dif <- abs(data$KLAASEN_1 - data$KLAASEN_2)

data$natur_adj_1 <- 8 - log(data$Team_1_rank)
data$natur_adj_2 <- 8 - log(data$Team_2_rank)
data$natur_dif <- abs(data$natur_adj_1 - data$natur_adj_2)

######## Making variable for who won ########

data$Map_1_team_1_point <- ifelse(data$Team_1_score_map_1 > data$Team_2_score_map_1, 1, 0)
data$Map_1_team_2_point <- ifelse(data$Map_1_team_1_point == 0, 1, 0)

data$Map_2_team_1_point <- ifelse(data$Team_1_score_map_2 > data$Team_2_score_map_2, 1, 0)
data$Map_2_team_2_point <- ifelse(data$Map_2_team_1_point == 0, 1, 0)

data$Map_3_team_1_point <- ifelse(data$Team_1_score_map_3 > data$Team_2_score_map_3, 1, 0)
data$Map_3_team_2_point <- ifelse(data$Map_3_team_1_point == 0, 1, 0)

data$Map_4_team_1_point <- ifelse(data$Team_1_score_map_4 > data$Team_2_score_map_4, 1, 0)
data$Map_4_team_2_point <- ifelse(data$Map_4_team_1_point == 0, 1, 0)

data$Map_5_team_1_point <- ifelse(data$Team_1_score_map_5 > data$Team_2_score_map_5, 1, 0)
data$Map_5_team_2_point <- ifelse(data$Map_5_team_1_point == 0, 1, 0)

data[is.na(data)] <- 0

data$team_1_mapwon <- data$Map_1_team_1_point + data$Map_2_team_1_point + data$Map_3_team_1_point + data$Map_4_team_1_point + data$Map_5_team_1_point
data$team_2_mapwon <- data$Map_1_team_2_point + data$Map_2_team_2_point + data$Map_3_team_2_point + data$Map_4_team_2_point + data$Map_5_team_2_point

data$team_1_winner <- ifelse(data$team_1_mapwon > data$team_2_mapwon, 1, 0)
data$team_2_winner <- ifelse(data$team_1_winner == 0, 1, 0)

data$strength <- abs(data$Team_2_rank - data$Team_1_rank)
data$strength2 <- data$strength^2

data$favourite_1 <- ifelse(data$Team_1_rank < data$Team_2_rank, 1, 0)
data$favourite_2 <- ifelse(data$Team_1_rank > data$Team_2_rank, 1, 0)
  
data$fav_won_1 <- ifelse(data$favourite_1*data$team_1_winner == 1, 1, 0)
data$fav_won_2 <- ifelse(data$favourite_2*data$team_2_winner == 1, 1, 0)
data$fav_won <- data$fav_won_1 + data$fav_won_2

nrow(data[data$Team_1_rank < 27 & data$Team_2_rank < 27,])/nrow(data)*100

################################################################ TENNIS ######################################################################

tennis <- read.csv("D:/Plocha/Doktor?t/HLTV/data_tennis.csv", header = TRUE, stringsAsFactors = FALSE, sep = ",")
tennis <- tennis[,-c(1, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17:24, 26:45, 47, 49)]
tennis <- na.omit(tennis)

tennis$KLAASEN_1 <- 8 - log2(tennis$winner_rank)
tennis$KLAASEN_2 <- 8 - log2(tennis$loser_rank)
tennis$KLAASEN_dif <- abs(tennis$KLAASEN_1 - tennis$KLAASEN_2)

tennis$natur_adj_1 <- 8 - log(tennis$winner_rank)
tennis$natur_adj_2 <- 8 - log(tennis$loser_rank)
tennis$natur_dif <- abs(tennis$natur_adj_1 - tennis$natur_adj_2)

tennis$fav_won <- ifelse(tennis$winner_rank < tennis$loser_rank, 1, 0)

cor.test(tennis$natur_dif, tennis$fav_won)

######## Logits ########

Logit <- glm(fav_won ~ strength, data = data, family = binomial)
Logit2 <- glm(fav_won ~ strength + strength2, data = data, family = binomial)
Logit3 <- glm(fav_won ~ strength + strength2 + Tier + strength:Tier, data = data, family = binomial)
Logit4 <- glm(fav_won ~ strength + strength2 + Type + strength:Type, data = data, family = binomial)
Logit5 <- glm(fav_won ~ strength + strength2 + Format_BO + strength:Format_BO, data = data, family = binomial)
Logit6 <- glm(fav_won ~ strength + strength2 + Stage + strength:Stage, data = data, family = binomial)

######## Summaries ########

summary(Logit)
summary(Logit2)
summary(Logit3)
summary(Logit4)
summary(Logit5)
summary(Logit6)

######## VIFs ########

VIF(Logit2)
VIF(Logit3)
VIF(Logit4)
VIF(Logit5)
VIF(Logit6)

BIC(Logit, Logit2, Logit3, Logit4, Logit5, Logit6)
stargazer(Logit, Logit2, Logit3, Logit4, Logit5, Logit6, star.cutoffs = c(0.05, 0.01, 0.001), type = "latex")

######## Correlations ########

favourite_win <- length(data$fav_won[data$fav_won == 1])
favourite_win
prob_fav_win <- favourite_win/782*100
prob_fav_win

cor.test(data$favourite_1, data$team_1_winner)
cor.test(data$favourite_2, data$team_2_winner)

(prob_fav_win-50)/50 - cor(data$favourite_1, data$team_1_winner)


cor.test(data$strength, data$fav_won)

cor.test(data$natur_dif, data$fav_won)

plot(data$natur_dif, data$fav_won)

summary(glm(data$favourite_1 ~ data$team_1_winner , data = data, family = binomial))

####

favourite_win_tennis <- length(tennis$fav_won[tennis$fav_won == 1])
favourite_win_tennis
prob_fav_win_tennis <- favourite_win_tennis/1418*100
prob_fav_win_tennis

cor.test(tennis$natur_dif, tennis$fav_won)

##############################################################################################################################################
################################################################ SUMMARY STATISTICS ##########################################################
##############################################################################################################################################

summary(data)
stargazer(data)
sd(data$Rank_Difference)
table(data$Type)
table(data$Tier)
table(data$Format_BO)

################################################################################################################################################################################################
################################################################################################################################################################################################
####################################################################################  KLAASEN  #################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################

######## Logits ########

Logit <- glm(fav_won ~ KLAASEN_dif, data = data, family = binomial)
Logit2 <- glm(fav_won ~ KLAASEN_dif + Tier + KLAASEN_dif:Tier, data = data, family = binomial)
Logit3 <- glm(fav_won ~ KLAASEN_dif + Type + KLAASEN_dif:Type, data = data, family = binomial)
Logit4 <- glm(fav_won ~ KLAASEN_dif + Format_BO + KLAASEN_dif:Format_BO, data = data, family = binomial)
Logit5 <- glm(fav_won ~ KLAASEN_dif + Stage + KLAASEN_dif:Stage, data = data, family = binomial)
Logit6 <- glm(fav_won ~ KLAASEN_dif + Tier + KLAASEN_dif:Tier + Type + KLAASEN_dif:Type + Format_BO + KLAASEN_dif:Format_BO + Stage + KLAASEN_dif:Stage, data = data, family = binomial)

######## Summaries ########

summary(Logit)
summary(Logit2)
summary(Logit3)
summary(Logit4)
summary(Logit5)
summary(Logit6)

######## VIFs ########

VIF(Logit2)
VIF(Logit3)
VIF(Logit4)
VIF(Logit5)
VIF(Logit6)

BIC(Logit, Logit2, Logit3, Logit4, Logit5, Logit6)
stargazer(Logit, Logit2, Logit3, Logit4, Logit5, Logit6, star.cutoffs = c(0.05, 0.01, 0.001), type = "latex")

################################################################ TENNIS ######################################################################

Logit_tennis <- glm(fav_won ~ natur_dif, data = tennis, family = binomial)

######## Summaries ########

summary(Logit_tennis)

######## VIFs ########

VIF(Logit_tennis)


alpha <- seq(0.2, 1, by = 0.01)
n1 = length(alpha)
g = matrix(nrow = n1, ncol = 4)
for (k in 1:length(alpha)){
  g[k,] <- gcv(fav_won ~ lp(natur_dif, nn = alpha[k], type = 4), family = 'binomial', data = data)
}
g

plot(g[,4]~g[,3])

LREG <- locfit(fav_won ~ lp(natur_dif, nn = 1, type = 4), family='binomial', data = data)

predLREG <- predict(LREG, newdata = data$natur_dif, band = "local")

cs_loc <- ggplot(data, aes(x = natur_dif, y = fav_won) ) +
  geom_line( aes(y = Logit$fitted.values), color = "black", size = .6) +
  ylim(.5, 1) +
  labs(x = "Skill difference", y = "") +
  theme(axis.title.x = element_text(size = 10, vjust = 0)) +
  theme(axis.line.y = element_line(color = "black")) +
  theme(axis.line.x = element_line(color = "black")) +
  theme(axis.text.y = element_text(colour = "black")) +
  theme(axis.text.x = element_text(colour = "black")) +
  theme(plot.margin = unit(c(.2, .2, .2, -.25), "cm"))

cs_log <- ggplot(data, aes(x = natur_dif, y = fav_won) ) +
  geom_line( aes(y = Logit$fitted.values), color = "black", size = .6) +
  ylim(.5, 1) +
  labs(x = "Skill difference", y = "Probability the favorite wins") +
  theme(axis.title.x = element_text(size = 10, vjust = 0)) +
  theme(axis.line.y = element_line(color = "black")) +
  theme(axis.line.x = element_line(color = "black")) +
  theme(axis.text.y = element_text(colour = "black")) +
  theme(axis.text.x = element_text(colour = "black"))



LREG_tennis <- locfit(fav_won ~ lp(natur_dif, nn = 1, type = 4), family='binomial', data = tennis)

predLREG_tennis <- predict(LREG_tennis, newdata = tennis$natur_dif, band = "local")

tennis_loc <- ggplot(tennis, aes(x = natur_dif, y = fav_won) ) +
  geom_line( aes(y = predLREG_tennis$fit,), color = "black", size = .6) +
  ylim(.5, 1) +
  xlim(0, 5) +
  labs(x = "Skill difference", y = "") +
  theme(axis.title.x = element_text(size = 10, vjust = 0)) +
  theme(axis.line.y = element_line(color = "black")) +
  theme(axis.line.x = element_line(color = "black")) +
  theme(axis.text.y = element_text(colour = "black")) +
  theme(axis.text.x = element_text(colour = "black")) +
  theme(plot.margin = unit(c(.2, .2, .2, -.25), "cm"))

tennis_log <- ggplot(tennis, aes(x = natur_dif, y = fav_won) ) +
  geom_line( aes(y = Logit_tennis$fitted.values), color = "black", size = .6) +
  ylim(.5, 1) +
  xlim(0, 5) +
  labs(x = "Skill difference", y = "Probability the favorite wins") +
  theme(axis.title.x = element_text(size = 10, vjust = 0)) +
  theme(axis.line.y = element_line(color = "black")) +
  theme(axis.line.x = element_line(color = "black")) +
  theme(axis.text.y = element_text(colour = "black")) +
  theme(axis.text.x = element_text(colour = "black"))

ggdraw(plot_grid(cs_log, cs_loc, tennis_log, tennis_loc), xlim = c(0, 1), ylim = c(0, 1), clip = "on") +
  draw_text(stringr::str_wrap("Logistic regression for Counter-Strike: Global Offensive", 30), x = 0.195, y = 0.93, size = 10) +
  draw_text(stringr::str_wrap("Logistic regression for Tennis", 20), x = 0.185, y = 0.43, size = 10) +
  draw_text(stringr::str_wrap("Local logistic regression for Counter-Strike: Global Offensive", 30), x = 0.715, y = 0.93, size = 10) +
  draw_text(stringr::str_wrap("Local logistic regression for Tennis", 25), x = 0.7, y = 0.43, size = 10) 


#### Uk?zka
graph %>%
  ggplot(aes(x = Prize)) +
  geom_histogram(color = "#929292", fill = "grey", bins = 30, alpha = 0.8) +
  scale_x_continuous(name = "Prize", breaks = c(0, 2.302585, 4.60517, 6.907755, 9.21034, 11.51293, 13.81551, 16.1181), labels = c("$0", "$10", "$100", "$1,000", "$10,000", "$100,000", "$1,000,000", "$10,000,000"), expand = c(0,0)) +
  scale_y_log10(name = "Frequency", labels = scales::comma, breaks = c(10, 100, 1000, 10000, 100000), expand = c(0,0)) +
  theme(axis.title.x = element_text(face = "bold", size = 16, vjust = 0)) +
  theme(axis.title.y = element_text(face = "bold", size = 16, vjust = -2)) +
  theme(axis.line.y = element_line(color = "black")) +
  theme(axis.line.x = element_line(color = "black")) +
  theme(axis.text.y = element_text(colour = "black")) +
  theme(axis.text.x = element_text(colour = "black")) +
  theme(legend.box = "horizontal")



######## Natural log - for review ########

summary(glm(fav_won ~ KLAASEN_dif, data = data, family = binomial))
Logit2 <- glm(fav_won ~ natur_dif, data = data, family = binomial)
Logit3 <- glm(fav_won ~ natur_dif + Tier + natur_dif:Tier, data = data, family = binomial)
Logit4 <- glm(fav_won ~ natur_dif + Type + natur_dif:Type, data = data, family = binomial)
Logit5 <- glm(fav_won ~ natur_dif + Format_BO + natur_dif:Format_BO, data = data, family = binomial)
Logit6 <- glm(fav_won ~ natur_dif + Stage + natur_dif:Stage, data = data, family = binomial)


Logit_tennis <- glm(fav_won ~ natur_dif, data = tennis, family = binomial)
summary(Logit_tennis)

######## Summaries ########

summary(Logit2)
summary(Logit3)
summary(Logit4)
summary(Logit5)
summary(Logit6)

######## VIFs ########

VIF(Logit2)
VIF(Logit3)
VIF(Logit4)
VIF(Logit5)
VIF(Logit6)

BIC(Logit2, Logit3, Logit4, Logit5, Logit6)
stargazer(Logit2, Logit3, Logit4, Logit5, Logit6, star.cutoffs = c(0.05, 0.01, 0.001), type = "latex")
