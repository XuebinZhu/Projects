                            #############################################
                            ### Mashable News Article Sharing Project ###
                            #############################################

rm(list=ls())

#load all the required library
library(factoextra)
library(usmap)
library(randomForest)
library(quantreg)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(MASS)
library(usmap)
library(glmnet)
library(MLmetrics)
library(scales)
library(plfm)
library(caTools)
library(psych)
library(VIM)

#set seed for spliting data set
set.seed(123)

###################
###Data Cleaning###
###################


#read the data
dat0 <- read.csv("OnlineNewsPopularity.csv", header = T)

#combine and create new variables
dat0$weekday <-  as.numeric(dat0$weekday_is_monday) * 1 + 
  as.numeric(dat0$weekday_is_tuesday) * 2 + 
  as.numeric(dat0$weekday_is_wednesday) * 3 + 
  as.numeric(dat0$weekday_is_thursday) * 4 + 
  as.numeric(dat0$weekday_is_friday) * 5 + 
  as.numeric(dat0$weekday_is_saturday) * 6 +
  as.numeric(dat0$weekday_is_sunday) * 7

dat0$channel <-  as.numeric(dat0$data_channel_is_lifestyle) * 1 + 
  as.numeric(dat0$data_channel_is_entertainment) * 2 + 
  as.numeric(dat0$data_channel_is_bus) * 3 + 
  as.numeric(dat0$data_channel_is_socmed) * 4 + 
  as.numeric(dat0$data_channel_is_tech) * 5 + 
  as.numeric(dat0$data_channel_is_world) * 6
dat0$channel <- (dat0$channel +1)

#rename the variables
dat0$weekday <- factor(dat0$weekday, levels = 1:7, labels = c("Mon", "Tues", "Wed", 
                                                              "Thurs", "Fri", "Sat", "Sun"))
dat0$channel <- factor(dat0$channel, levels = 1:7, labels = c("Other", "Lifestyle", "Entertainment", "Business", 
                                                              "Social_Media", "Technology", "World"))
#use log transformation for shares
hist(dat0$shares)
dat0$log_shares <- log(dat0$shares)
hist(dat0$log_shares)

#drop meaningless variables
drop <- c("url","n_tokens_content", "n_unique_tokens", "num_hrefs", " kw_min_min", "kw_max_min","kw_avg_min", 
          "kw_min_max","kw_max_max", "kw_avg_max", "kw_min_avg"," kw_max_avg", "kw_avg_avg", "self_reference_min_shares", 
          "self_reference_max_shares", "abs_title_subjectivity", "abs_title_sentiment_polarity", "min_positive_polarity",
          "max_positive_polarity", "min_negative_polarity", "max_negative_polarity", "rate_positive_words", "rate_negative_words", 
          "kw_min_min", "kw_max_avg", "weekday_is_monday","weekday_is_tuesday" , "weekday_is_wednesday", "weekday_is_thursday", 
          "weekday_is_friday","weekday_is_saturday","weekday_is_sunday", "data_channel_is_lifestyle", "data_channel_is_entertainment", 
          "data_channel_is_bus","data_channel_is_socmed", "data_channel_is_tech","data_channel_is_world", "shares", 
          "LDA_00", "LDA_01", "LDA_02", "LDA_03", "LDA_04")

dat0 <- dat0[,!(names(dat0) %in% drop)]

#convert data type
dat0$is_weekend <- as.factor(dat0$is_weekend)

#create full data set for this project
news <- dat0

#remove outliers
n_non_stop_words_Q1SP <- summary(news$n_non_stop_words)[2]
n_non_stop_words_Q3SP <- summary(news$n_non_stop_words)[5]
n_non_stop_words_IQRSP <- n_non_stop_words_Q3SP - n_non_stop_words_Q1SP
min_n_non_stop_words <- as.numeric(n_non_stop_words_Q1SP - 1.5*n_non_stop_words_IQRSP)
max_n_non_stop_words <- as.numeric(n_non_stop_words_Q3SP + 1.5*n_non_stop_words_IQRSP)
news <- news %>% filter(n_non_stop_words < max_n_non_stop_words)
news <- news %>% filter(n_non_stop_words > min_n_non_stop_words)

summary(dat0$n_non_stop_words)
boxplot(news$n_non_stop_words)
hist(news$n_non_stop_words)#this variable seems to have lots of errors, so I decided to remove it

#"is_weekend" is highly co-related with weekday, so I decided to remove it
#"timedelta" gives no predictive information for our y variable, so I decided to remove it

news <- dat0
drop1 <- c("n_non_stop_words", "is_weekend", "timedelta")
news <- news[,!(names(news) %in% drop1)]

#remove outliers
n_non_stop_unique_tokens_Q1SP <- summary(news$n_non_stop_unique_tokens)[2]
n_non_stop_unique_tokens_Q3SP <- summary(news$n_non_stop_unique_tokens)[5]
n_non_stop_unique_tokens_IQRSP <- n_non_stop_unique_tokens_Q3SP - n_non_stop_unique_tokens_Q1SP
min_n_non_stop_unique_tokens <- as.numeric(n_non_stop_unique_tokens_Q1SP - 1.5*n_non_stop_unique_tokens_IQRSP)
max_n_non_stop_unique_tokens <- as.numeric(n_non_stop_unique_tokens_Q3SP + 1.5*n_non_stop_unique_tokens_IQRSP)
news <- news %>% filter(n_non_stop_unique_tokens < max_n_non_stop_unique_tokens)
news <- news %>% filter(n_non_stop_unique_tokens > min_n_non_stop_unique_tokens)

summary(news$n_non_stop_unique_tokens)
boxplot(news$n_non_stop_unique_tokens)
hist(news$n_non_stop_unique_tokens)#keep

#num_self_hrefs, num_imgs, num_videos, average_token_length, self_reference_avg_sharess should do log transformation
news$num_self_hrefs <- log(1 + news$num_self_hrefs)
news$num_imgs <- log(1 + news$num_imgs)
news$num_videos <- log(1 + news$num_videos)
news$average_token_length <- log(1 + news$average_token_length)
news$self_reference_avg_sharess <- log(1 + news$self_reference_avg_sharess)

summary(news)

#impute NA value in Channel Column
sum(is.na(news$channel))
summary(news$channel)
# Get levels and add "Unknown"
levels <- levels(news$channel)
levels[length(levels) + 1] <- "Unknown"
# refactor to include "Unknown" as a factor level
# and replace NA with "Unknown"
news$channel <- factor(news$channel, levels = levels)
news$channel[is.na(news$channel)] <- "Unknown"

#split the data
split <- sample.split(news, SplitRatio = 0.8)
data_train <- subset(news, split==T)
data_test <- subset(news, split==F)

#the final train data set
head(data_train)
#the final test data set
head(data_test)


########################
###Data Visualization###
########################

ggplot(data_train, aes(x = weekday, fill = channel)) +
  geom_bar(show.legend = T) +
  theme_minimal() +
  labs(title = 'Number of Articles in Different Weekday', 
       subtitle = 'divided by different channel',
       x = 'Weekday', y = 'Number of Articles') +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))

ggplot(data_train, aes(x = channel, fill = channel)) +
  geom_bar(show.legend = F) +
  theme_minimal() +
  labs(title = 'Number of Articles in Different Channel',
       x = 'Channel', y = 'Number of Articles') + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  theme(axis.text.x = element_text(angle=30, hjust=1, vjust=1))

ggplot(dat0, aes(x = channel, y = log_shares, fill = as.factor(is_weekend))) +
  geom_boxplot(show.legend = T) +
  theme_minimal() +
  labs(title = 'Log Number of Shares versus Different Channel', 
       subtitle = 'divided by weekday and weekend',
       x = 'Channel', y = 'Log Number of Shares') + 
  scale_fill_discrete(name="Weekday or Weekend",
                      breaks=c("0", "1"),
                      labels=c("Weekday", "Weekend"))+
  scale_x_discrete(labels=c("Other", "Lifestyle", "Entertainment", "Business", "Social Media", "Technology", "World")) +
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))+ 
  theme(axis.text.x = element_text(angle=30, hjust=1, vjust=1))

ggplot(data_train, aes(x = global_subjectivity, y = log_shares, color = channel)) +
  geom_point(show.legend = T) +
  theme_minimal() +
  labs(title = 'Log Number of Shares versus Article Subjectivity', 
       subtitle = 'divided by different channel',
       x = 'Article Subjectivity', y = 'Log Number of Shares') + 
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))

ggplot(data_train, aes(x = global_sentiment_polarity, y = log_shares, color = channel)) +
  geom_point(show.legend = T) +
  theme_minimal() +
  labs(title = 'Log Number of Shares versus Article Sentiment Polarity', 
       subtitle = 'divided by different channel',
       x = 'Article Sentiment Polarity', y = 'Log Number of Shares') + 
  theme(plot.title = element_text(hjust = 0.5), plot.subtitle = element_text(hjust = 1))


##############
###Modeling###
##############

#Core Task: Unsupervised Learning  
#Data Mining Method: PCA and K-Means Clustering  

#PCA
#Computing the full PCA
clustering_set <- data_train[,1:16]
clustering_set <- scale(clustering_set)
#Computing the full PCA
pca.news <- prcomp(clustering_set, scale=TRUE)
###Plotting the variance that each component explains
plot(pca.news,main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)
screeplot(pca.news,type = "line", main = "Scree Plot for PCA", npcs = length(pca.news$sdev))
mtext(side=3, "Number of Factors",  line= -18, font=2)
abline(v = 5)

#get the scores and the loadings of the five selected PC
pcScore <- predict(pca.news)
pcScore_selected <- pcScore[,1:2]
pcScore_selected2 <- pcScore[,3:4]
loadings <- pca.news$rotation[,1:5]
nrow(loadings)

#### PC1
# Positive sentiment articles
v1 <- loadings[order(abs(loadings[,1]), decreasing=TRUE)[1:19],1]
loadingfit1 <- lapply(1:27, function(k) ( t(v1[1:k])%*%v1[1:k] - 3/4 )^2)
v1[1:which.min(loadingfit1)]
#### PC2
# Negative Sentiment aritcles
v2 <- loadings[order(abs(loadings[,2]), decreasing=TRUE)[1:19],2]
loadingfit2 <- lapply(1:27, function(k) ( t(v2[1:k])%*%v2[1:k] - 3/4 )^2)
v2[1:which.min(loadingfit2)]
#### PC3
# short articles with more links and images
v3 <- loadings[order(abs(loadings[,3]), decreasing=TRUE)[1:19],3]
loadingfit3 <- lapply(1:27, function(k) ( t(v3[1:k])%*%v3[1:k] - 3/4 )^2)
v3[1:which.min(loadingfit3)]
#### PC4
# Long articles Less image but use more popular links
v4 <- loadings[order(abs(loadings[,4]), decreasing=TRUE)[1:19],4]
loadingfit4 <- lapply(1:27, function(k) ( t(v4[1:k])%*%v4[1:k] - 3/4 )^2)
v4[1:which.min(loadingfit4)]
#### PC5
v5 <- loadings[order(abs(loadings[,4]), decreasing=TRUE)[1:19],5]
loadingfit5 <- lapply(1:27, function(k) ( t(v5[1:k])%*%v5[1:k] - 3/4 )^2)
v5[1:which.min(loadingfit5)]

#K-mean clustering
SpcScore3 <- scale(pcScore, scale=TRUE)
kfit <- lapply(1:30, function(k) kmeans(SpcScore3,k,nstart=10))

## Selects the Number of Clusters via an Information Criteria
## get AIC (option "A"), BIC (option "B"), HDIC (option "C") for the output of kmeans
kIC <- function(fit, rule=c("A","B","C")){
  df <- length(fit$centers) # K*dim
  n <- sum(fit$size)
  D <- fit$tot.withinss # deviance
  rule=match.arg(rule)
  if(rule=="A")
    #return(D + 2*df*n/max(1,n-df-1))
    return(D + 2*df)
  else if(rule=="B") 
    return(D + log(n)*df)
  else 
    return(D +  sqrt( n * log(df) )*df)
}

kaic <- sapply(kfit,kIC)
kbic  <- sapply(kfit,kIC,"B")
kcic  <- sapply(kfit,kIC,"C")
SpcScore<- scale(pcScore_selected, scale=TRUE)
K_clustering <- kmeans(as.data.frame(SpcScore),which.min(kcic),nstart=10)
K_clustering
fviz_cluster(K_clustering, data = as.data.frame(SpcScore), geom = "point", 
             show.clust.cent = FALSE, ellipse = FALSE,palette = "Set5", main = "K-means clustering with PC1 and PC2")

#Core Task: Supervised Learning: Regression  
#Data Mining Method: Linear Regression, Stepwise Selection, CART, Random Forest, Lasso, Post Lasso 

#full model
full_fit <- lm(log_shares ~ . , data = data_train)
summary(full_fit)
AIC(full_fit)

#full model with interaction
fit_interaction <- lm(log_shares ~ .^2, data = data_train)
summary(fit_interaction)
AIC(fit_interaction)

#stepwise regression
step(full_fit, trace = 1, direction = "both")
step_fit <- lm(log_shares ~ n_non_stop_unique_tokens + num_self_hrefs + 
                 num_imgs + num_videos + num_keywords + self_reference_avg_sharess + 
                 global_subjectivity + global_rate_negative_words + avg_positive_polarity + 
                 avg_negative_polarity + title_subjectivity + title_sentiment_polarity + 
                 weekday + channel, data = data_train)
summary(step_fit)
AIC(step_fit)

#PCA
pca_y <- data_train$log_shares
pca_x <- data.frame(pcScore)[,1:5]
pca_fit <- lm(pca_y ~ ., data = pca_x)
summary(pca_fit)

#CART
cart_fit <- rpart(log_shares ~ . , data = data_train, method = "anova", control = rpart.control(cp = .0001))
bestcp <- cart_fit$cptable[which.min(cart_fit$cptable[,"xerror"]),"CP"]
best_cart_fit <- prune(cart_fit, cp = bestcp)
rpart.plot(best_cart_fit, type = 3, digits = 3, fallen.leaves = T)
printcp(best_cart_fit)
plotcp(best_cart_fit)
tot_count <- function(x, labs, digits, varlen) {paste(labs, "\n\nn = ", x$frame$n)}
prp(best_cart_fit, faclen = 0, cex = 0.8, node.fun = tot_count)
mtext(side=1, "Weekday: 1 = Mon, 2 = Tues, \n 3 = Wed, 4 = Thurs, 5 = Fri", line = 3, font=1, adj = 1)
summary(best_cart_fit)

#Random Forest
forest_fit <- randomForest(log_shares ~ ., data = data_train, ntree=495, na.action=na.roughfix)
#which.min(regForest$mse)

#lasso
model_train_matrix <- model.matrix(log_shares ~ ., data = data_train)
model_train_y <- data_train$log_shares

#lasso selection
lasso_model1 <- glmnet(model_train_matrix, model_train_y, alpha = 1, family = 'gaussian')
plot(lasso_model1, xvar = 'lambda', label = T)
lasso_model2 <- cv.glmnet(model_train_matrix, model_train_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
lasso_lam_1se <- lasso_model2$lambda.1se
lasso_coef_1se <- coef(lasso_model1, lasso_lam_1se)
summary(lasso_coef_1se)
plot(lasso_model2)

lasso_model_best <- glmnet(model_train_matrix, model_train_y, alpha = 1, family = 'gaussian', lambda = lasso_lam_1se)
### Returns the indices for which |x[i]| > tr
support<- function(x, tr = 10e-6) {
  m<- rep(0, length(x))
  for (i in 1:length(x)) if( abs(x[i])> tr ) m[i]<- i
  m <- m[m>0]
  m
}
## get the support
lasso_supp <- support(lasso_model_best$beta)
length(lasso_supp)
colnames(model_train_matrix[,lasso_supp])

inthemodel_train <- unique(c(lasso_supp))# unique grabs union
lasso_train_data <- cBind(model_train_matrix[,inthemodel_train]) 
lasso_train_data <- as.data.frame(as.matrix(lasso_train_data))# make it a data.frame
dim(lasso_train_data) ## p about half n
#fit the final model
lasso_fit <- lm(model_train_y ~., data=lasso_train_data)
summary(lasso_fit)
AIC(lasso_fit)
#post lasso
pl_train_matrix <- model.matrix(log_shares ~ ., data = data_train)
pl_train_y <- data_train$log_shares
pl_model1 <- glmnet(pl_train_matrix, pl_train_y, alpha = 1, family = 'gaussian')
plot(pl_model1, xvar = 'lambda', label = T)
pl_model2 <- cv.glmnet(pl_train_matrix, pl_train_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
pl_lam_1se <- pl_model2$lambda.1se
pl_coef_1se <- coef(pl_model1, pl_lam_1se)
summary(pl_coef_1se)
plot(pl_model2)
pl_model_best <- glmnet(pl_train_matrix, pl_train_y, alpha = 1, family = 'gaussian', lambda = pl_lam_1se)

## get the support
pl_supp <- support(pl_model_best$beta)
length(pl_supp)
colnames(pl_train_matrix[,pl_supp])
data_pl_1se <- data.frame(pl_train_matrix[,pl_supp],pl_train_y)
#fit the pl
post_lasso <- lm(pl_train_y~., data=data_pl_1se)
summary(post_lasso)
AIC(post_lasso)


################
###Prediction###
################

MAE <- function(actual, predicted) {mean(abs(actual - predicted))}
R2 <- function(actual, predicted) {1 - (sum((predicted - actual)^2))/(sum((mean(actual) - actual)^2))}

test_with_prediction <- data_test
#Full
test_with_prediction$pred_full <- predict.lm(full_fit, newdata = test_with_prediction)
#Interaction
test_with_prediction$pred_interaction <- predict.lm(fit_interaction, newdata = test_with_prediction)
#Stepwise
test_with_prediction$pred_step <- predict.lm(step_fit, newdata = test_with_prediction)
#PCA
pca_test <- data_test[,1:16]
pca_test <- scale(pca_test)
#Computing the full PCA
pca_test_fit <- prcomp(pca_test, scale=TRUE)
#build PCA model
pca_test_scores <- predict(pca_test_fit)
pca_test_scores
pca_test_loadings <- pca_test_fit$rotation[,1:5]
pca_test_loadings
pca_test_y <- data_test$log_shares
pca_test_x <- data.frame(pca_test_scores)[,1:5]
pca_test_lmfit <- lm(pca_test_y ~ ., data = pca_test_x)
test_with_prediction$pred_pca <- predict(pca_test_lmfit, data = pca_test_x)
#CART
test_with_prediction$pred_cart <- predict(best_cart_fit, newdata = data_test, type="vector")
#Random Forest
test_with_prediction$pred_forest <- predict(forest_fit, newdata = data_test, type="response")
#Lasso
model_test <- data_test
model_test_matrix <- model.matrix( ~ ., data=model_test)
colnames(model_test_matrix[,lasso_supp])
inthemodel_test <- unique(c(lasso_supp))# unique grabs union
lasso_test_data <- cBind(model_test_matrix[,inthemodel_test]) 
lasso_test_data <- as.data.frame(as.matrix(lasso_test_data))# make it a data.frame
dim(lasso_test_data) ## p about half n
lasso_pred <- predict.lm(lasso_fit, newdata = lasso_test_data,type="response")
lasso_pred
model_test_with_prediction <- model_test
model_test_with_prediction$lasso_pred <- lasso_pred
test_with_prediction$pred_lasso <- predict.lm(lasso_fit, newdata = lasso_test_data,type="response")
#Post Lasso
pl_model_test <- data_test
pl_model_test_matrix <- model.matrix( ~ ., data=model_test)
colnames(pl_model_test_matrix[,pl_supp])
pl_inthemodel_test <- unique(c(pl_supp))# unique grabs union
pl_test_data <- cBind(pl_model_test_matrix[,pl_inthemodel_test]) 
pl_test_data <- as.data.frame(as.matrix(pl_test_data))# make it a data.frame
dim(pl_test_data) ## p about half n
test_with_prediction$pred_pl <- predict.lm(post_lasso, newdata = pl_test_data,type="response")


################
###Evaluation###
################

summary(full_fit)
aic_full <- AIC(full_fit)
coef_full <- data.frame(full_fit$coefficients)
ncoef_full <- nrow(coef_full)
mae_full <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_full)
mse_full <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_full)
R2_full <- R2(test_with_prediction$log_shares, test_with_prediction$pred_full)

summary(fit_interaction)
coef_interaction <- data.frame(fit_interaction$coefficients)
ncoef_interaction <- nrow(coef_interaction)
aic_interaction <- AIC(fit_interaction)
mae_interaction <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_interaction)
mse_interaction <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_interaction)
R2_interaction <- R2(test_with_prediction$log_shares, test_with_prediction$pred_interaction)

summary(step_fit)
coef_step <- data.frame(step_fit$coefficients)
ncoef_step <- nrow(coef_step)
aic_step <- AIC(step_fit)
mae_step <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_step)
mse_step <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_step)
R2_step <- R2(test_with_prediction$log_shares, test_with_prediction$pred_step)

summary(pca_fit)
coef_pca <- data.frame(pca_fit$coefficients)
ncoef_pca <- nrow(coef_pca)
aic_pca <- AIC(pca_fit) 
mae_pca <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_pca)
mse_pca <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_pca)
R2_pca <- R2(test_with_prediction$log_shares, test_with_prediction$pred_pca)

summary(best_cart_fit)
coef_cart <- data.frame(best_cart_fit$cptable)
ncoef_cart <- nrow(coef_cart)
aic_cart <- AIC(best_cart_fit) #no aic
mae_cart <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_cart)
mse_cart <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_cart)
R2_cart <- R2(test_with_prediction$log_shares, test_with_prediction$pred_cart)
summary(dat0$global_subjectivity)

coef_forest <- data.frame(forest_fit$cptable)#can't make it a data frame
ncoef_forest <- forest_fit$ntree
aic_forest <- AIC(forest_fit) #no aic
mae_forest <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_forest)
mse_forest <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_forest)
R2_forest <- R2(test_with_prediction$log_shares, test_with_prediction$pred_forest)

summary(lasso_fit)
coef_lasso <- data.frame(lasso_fit$coefficients)
ncoef_lasso <- nrow(coef_lasso)
aic_lasso <- AIC(lasso_fit) 
mae_lasso <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_lasso)
mse_lasso <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_lasso)
R2_lasso <- R2(test_with_prediction$log_shares, test_with_prediction$pred_lasso)

summary(post_lasso)
coef_pl <- data.frame(post_lasso$coefficients)
ncoef_pl <- nrow(coef_pl)
aic_pl <- AIC(post_lasso)
mae_pl <- MAE(test_with_prediction$log_shares, test_with_prediction$pred_pl)
mse_pl <- MSE(test_with_prediction$log_shares, test_with_prediction$pred_pl)
R2_pl <- R2(test_with_prediction$log_shares, test_with_prediction$pred_pl)

#create evaluation metrics data frame
mae_data <- data.frame(cbind(mae_full, mae_interaction, mae_step, mae_pca, mae_cart, mae_forest, mae_lasso, mae_pl))
mse_data <- data.frame(cbind(mse_full, mse_interaction, mse_step, mse_pca, mse_cart, mse_forest, mse_lasso, mse_pl))
aic_data <- data.frame(cbind(aic_full, aic_interaction, aic_step, aic_pca, aic_lasso, aic_pl))
R2_data <- data.frame(cbind(R2_full, R2_interaction, R2_step, R2_pca, R2_cart, R2_forest, R2_lasso, R2_pl))
num_coef <- data.frame(cbind(ncoef_full, ncoef_interaction, ncoef_step, ncoef_pca, ncoef_cart, ncoef_forest, ncoef_lasso, ncoef_pl))

#visualize OOS performance
barplot(height = cbind(mae_full, mae_interaction, mae_step, mae_pca, mae_cart, mae_forest, mae_lasso, mae_pl), space = 0.1, ylim =c(0, 0.8), 
        names.arg = c("Full", "INT", "Step", "PCA", "CART", "Forest", "Lasso", "PL"), ylab = "OOS Mean Absolute Error",
        main = "OOS MAE Across Different Models")
abline(h = sum(mae_data[1,])/ncol(mae_data), col = "red", lwd = 2)
mtext(side=1, "Line represents the average", col = "red", line = 4, font=1, adj = 1)

barplot(height = cbind(mse_full, mse_interaction, mse_step, mse_pca, mse_cart, mae_forest, mse_lasso, mse_pl), space = 0.1, ylim =c(0, 0.8), 
        names.arg = c("Full", "INT", "Step", "PCA", "CART", "Forest", "Lasso", "PL"), ylab = "OOS Mean Square Error",
        main = "OOS MSE Across Different Models")
abline(h = sum(mse_data[1,])/ncol(mse_data), col = "red", lwd = 2)
mtext(side=1, "Line represents the average", col = "red", line = 4, font=1, adj = 1)

barplot(height = cbind(aic_full, aic_interaction, aic_step, aic_pca, aic_lasso, aic_pl), space = 0.1, 
        names.arg = c("Full", "INT", "Step", "PCA", "Lasso", "PL"), ylab = "AIC Value", ylim =c(0, 80000),
        main = "AIC Values Across Different Models")
abline(h = sum(aic_data[1,])/ncol(aic_data), col = "red", lwd = 2)

barplot(height = cbind(R2_full, R2_interaction, R2_step, R2_pca, R2_cart, R2_forest, R2_lasso, R2_pl), space = 0.1, ylim =c(0, 0.14), 
        names.arg = c("Full", "INT", "Step", "PCA", "CART", "Forest", "Lasso", "PL"), ylab = "OOS R Square",
        main = "OOS R2 Across Different Models")
abline(h = sum(R2_data[1,])/ncol(R2_data), col = "red", lwd = 2)
mtext(side=1, "Line represents the average", col = "red", line = 4, font=1, adj = 1)

barplot(height = cbind(ncoef_full, ncoef_interaction, ncoef_step, ncoef_pca, ncoef_cart, ncoef_forest, ncoef_lasso, ncoef_pl), space = 0.1, ylim =c(0, 500), 
        names.arg = c("Full", "INT", "Step", "PCA", "CART", "Forest", "Lasso", "PL"), ylab = "Number of Coefficients",
        main = "Number of Coefficients Across Different Models")
abline(h = sum(num_coef[1,])/ncol(num_coef), col = "red", lwd = 2)
mtext(side=1, " Number of splits for CART \n Number of trees for Random Forest", line = 4, font=1, adj = 0)
mtext(side=1, "Line represents the average", col = "red", line = 4, font=1, adj = 1)

barplot(height = cbind(ncoef_full, ncoef_step, ncoef_pca, ncoef_cart, ncoef_lasso, ncoef_pl), space = 0.1, ylim =c(0, 30), 
        names.arg = c("Full", "Step", "PCA", "CART", "Lasso", "PL"), ylab = "Number of Coefficients",
        main = "Number of Coefficients Across Different Models")
abline(h = (sum(num_coef[1,])-num_coef[1,2]-num_coef[1,6])/(ncol(num_coef)-2), col = "red", lwd = 2)


#########################
###Quantile Regression###
#########################

data_quan <- lasso_train_data
data_quan$log_shares <- data_train$log_shares
### Look at the lower tail say tau = 0.05
lower.tail <- rq(log_shares ~ ., tau = 0.05, data = data_quan) 
coef(lower.tail)
### Look at the upper tail say tau = 0.95
upper.tail <- rq(log_shares ~ ., tau = 0.95, data = data_quan) 
coef(upper.tail)

###taus from 0.05 to 0.95
taus <- seq(from=0.05,to=0.95,by=0.05)

rq_taus <-rq(log_shares ~ ., tau = taus, data = data_quan)
fittaus_rq <- summary(rq_taus)

#visualize the different variables' effects across different taus
plot(fittaus_rq)
mtext("Values of taus", side=1, line = 4.2, font=1, col = 'blue')
mtext("Values of coefficients", side=2, line = 3.3, font=1, col = 'blue')

plot(fittaus_rq, parm = 1, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for Intercept ")
plot(fittaus_rq, parm = 2, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for num_self_hrefs ")
plot(fittaus_rq, parm = 3, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for num_imgs ")
plot(fittaus_rq, parm = 4, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for num_videos ")
plot(fittaus_rq, parm = 5, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for num_keywords ")
plot(fittaus_rq, parm = 6, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for self_reference_avg_sharess ")
plot(fittaus_rq, parm = 7, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for global_subjectivity ")
plot(fittaus_rq, parm = 8, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for avg_negative_polarity ")
plot(fittaus_rq, parm = 9, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for title_subjectivity ")
plot(fittaus_rq, parm = 10, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for title_sentiment_polarity ")
plot(fittaus_rq, parm = 11, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for Saturday ")
plot(fittaus_rq, parm = 12, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for Sunday ")
plot(fittaus_rq, parm = 13, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for Entertainment ")
plot(fittaus_rq, parm = 14, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for Business ")
plot(fittaus_rq, parm = 15, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for Social_Media ")
plot(fittaus_rq, parm = 16, xlab = "Quantile Index", ylab = "Coefficient Value", main = "Coefficient for World ")


###############
###causality###
###############

#####
##### Double Selection Procedure for Robust confidence Intervals
##### 3 steps
##### 1. run Lasso of Y on controls X
##### 2. run Lasso of d on controls X
##### 3. Take all selectedX (Step 1 and 2) and run Linear regression of
#####         Y on treatment d and selectedX
#####

causality_data_train <- lasso_train_data
causality_y <- data_train$log_shares
### treatment is num_imgs
causality_x_num_imgs <- model.matrix(causality_y ~ .-num_imgs, data = causality_data_train )
treatment_num_imgs <- causality_data_train$num_imgs

## Step 1.Select controls that are good to predict the outcome
causality_model1_num_imgs <- cv.glmnet(causality_x_num_imgs, causality_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_1se_num_imgs <- causality_model1_num_imgs$lambda.1se
# Call Lasso 
causality_lasso_1se_num_imgs <- glmnet(causality_x_num_imgs, causality_y,alpha = 1, family = 'gaussian',lambda = causality_lam_1se_num_imgs)
# Get the support
causality_supp1_num_imgs <- support(causality_lasso_1se_num_imgs$beta)
# Step 1 selected
length(causality_supp1_num_imgs)
### controls
colnames(causality_x_num_imgs[,causality_supp1_num_imgs])

###
### Step 2.Select controls that are good to predict the treatment
causality_model2_num_imgs <- cv.glmnet(causality_x_num_imgs, treatment_num_imgs, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_d_1se_num_imgs <- causality_model2_num_imgs$lambda.1se
# Call Lasso 
causality_lam_d_1se_num_imgs <- glmnet(causality_x_num_imgs, treatment_num_imgs, alpha = 1, family = 'gaussian',lambda = causality_lam_d_1se_num_imgs)
# Get the support
causality_supp2_num_imgs <-support(causality_lam_d_1se_num_imgs$beta)
### Step 2 selected
length(causality_supp2_num_imgs)
### controls
colnames(causality_x_num_imgs[,causality_supp2_num_imgs])

###
### Step 3.Combine all selected and refit linear regression
causality_inthemodel_num_imgs <- unique(c(causality_supp1_num_imgs, causality_supp2_num_imgs)) # unique grabs union
selectdata_num_imgs <- cBind(treatment_num_imgs, causality_x_num_imgs[,causality_inthemodel_num_imgs]) 
selectdata_num_imgs <- as.data.frame(as.matrix(selectdata_num_imgs)) # make it a data.frame
dim(selectdata_num_imgs) ## p about half n

causality_lm_num_imgs <- lm(causality_y ~ ., data = selectdata_num_imgs)
summary(causality_lm_num_imgs)$coef["treatment_num_imgs",]

####
#### Compare/contrast the results with the following models
#### 
simple_num_imgs <- lm(causality_y ~ num_imgs, data = causality_data_train)
summary(simple_num_imgs)
summary(full_fit)
summary(step_fit)
summary(lasso_fit)
summary(post_lasso)
coef(lower.tail)
coef(upper.tail)


### treatment is num_videos
causality_x_num_videos <- model.matrix(causality_y ~ .-num_videos, data = causality_data_train )
treatment_num_videos <- causality_data_train$num_videos

## Step 1.Select controls that are good to predict the outcome
causality_model1_num_videos <- cv.glmnet(causality_x_num_videos, causality_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_1se_num_videos <- causality_model1_num_videos$lambda.1se
# Call Lasso 
causality_lasso_1se_num_videos <- glmnet(causality_x_num_videos, causality_y,alpha = 1, family = 'gaussian',lambda = causality_lam_1se_num_videos)
# Get the support
causality_supp1_num_videos <- support(causality_lasso_1se_num_videos$beta)
# Step 1 selected
length(causality_supp1_num_videos)
### controls
colnames(causality_x_num_videos[,causality_supp1_num_videos])

###
### Step 2.Select controls that are good to predict the treatment
causality_model2_num_videos <- cv.glmnet(causality_x_num_videos, treatment_num_videos, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_d_1se_num_videos <- causality_model2_num_videos$lambda.1se
# Call Lasso 
causality_lam_d_1se_num_videos <- glmnet(causality_x_num_videos, treatment_num_videos, alpha = 1, family = 'gaussian',lambda = causality_lam_d_1se_num_videos)
# Get the support
causality_supp2_num_videos <-support(causality_lam_d_1se_num_videos$beta)
### Step 2 selected
length(causality_supp2_num_videos)
### controls
colnames(causality_x_num_videos[,causality_supp2_num_videos])

###
### Step 3.Combine all selected and refit linear regression
causality_inthemodel_num_videos <- unique(c(causality_supp1_num_videos, causality_supp2_num_videos)) # unique grabs union
selectdata_num_videos <- cBind(treatment_num_videos, causality_x_num_videos[,causality_inthemodel_num_videos]) 
selectdata_num_videos <- as.data.frame(as.matrix(selectdata_num_videos)) # make it a data.frame
dim(selectdata_num_videos) ## p about half n

causality_lm_num_videos <- lm(causality_y ~ ., data = selectdata_num_videos)
summary(causality_lm_num_videos)$coef["treatment_num_videos",]

####
#### Compare/contrast the results with the following models
#### 
simple_num_videos <- lm(causality_y ~ num_videos, data = causality_data_train)
summary(simple_num_videos)
summary(full_fit)
summary(step_fit)
summary(lasso_fit)
summary(post_lasso)
coef(lower.tail)
coef(upper.tail)

### treatment is global_subjectivity
causality_x_global_subjectivity <- model.matrix(causality_y ~ .-global_subjectivity, data = causality_data_train )
treatment_global_subjectivity <- causality_data_train$global_subjectivity

## Step 1.Select controls that are good to predict the outcome
causality_model1_global_subjectivity <- cv.glmnet(causality_x_global_subjectivity, causality_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_1se_global_subjectivity <- causality_model1_global_subjectivity$lambda.1se
# Call Lasso 
causality_lasso_1se_global_subjectivity <- glmnet(causality_x_global_subjectivity, causality_y,alpha = 1, family = 'gaussian',lambda = causality_lam_1se_global_subjectivity)
# Get the support
causality_supp1_global_subjectivity <- support(causality_lasso_1se_global_subjectivity$beta)
# Step 1 selected
length(causality_supp1_global_subjectivity)
### controls
colnames(causality_x_global_subjectivity[,causality_supp1_global_subjectivity])

###
### Step 2.Select controls that are good to predict the treatment
causality_model2_global_subjectivity <- cv.glmnet(causality_x_global_subjectivity, treatment_global_subjectivity, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_d_1se_global_subjectivity <- causality_model2_global_subjectivity$lambda.1se
# Call Lasso 
causality_lam_d_1se_global_subjectivity <- glmnet(causality_x_global_subjectivity, treatment_global_subjectivity, alpha = 1, family = 'gaussian',lambda = causality_lam_d_1se_global_subjectivity)
# Get the support
causality_supp2_global_subjectivity <-support(causality_lam_d_1se_global_subjectivity$beta)
### Step 2 selected
length(causality_supp2_global_subjectivity)
### controls
colnames(causality_x_global_subjectivity[,causality_supp2_global_subjectivity])

###
### Step 3.Combine all selected and refit linear regression
causality_inthemodel_global_subjectivity <- unique(c(causality_supp1_global_subjectivity, causality_supp2_global_subjectivity)) # unique grabs union
selectdata_global_subjectivity <- cBind(treatment_global_subjectivity, causality_x_global_subjectivity[,causality_inthemodel_global_subjectivity]) 
selectdata_global_subjectivity <- as.data.frame(as.matrix(selectdata_global_subjectivity)) # make it a data.frame
dim(selectdata_global_subjectivity) ## p about half n

causality_lm_global_subjectivity <- lm(causality_y ~ ., data = selectdata_global_subjectivity)
summary(causality_lm_global_subjectivity)$coef["treatment_global_subjectivity",]

####
#### Feel free to compare/contrast your results with the following models
#### 
simple_global_subjectivity <- lm(causality_y ~ global_subjectivity, data = causality_data_train)
summary(simple_global_subjectivity)
summary(full_fit)
summary(step_fit)
summary(lasso_fit)
summary(post_lasso)
coef(lower.tail)
coef(upper.tail)

### treatment is Sunday
causality_x_Sunday <- model.matrix(causality_y ~ .-weekdaySun, data = causality_data_train )
treatment_Sunday <- causality_data_train$weekdaySun

## Step 1.Select controls that are good to predict the outcome
causality_model1_Sunday <- cv.glmnet(causality_x_Sunday, causality_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_1se_Sunday <- causality_model1_Sunday$lambda.1se
# Call Lasso 
causality_lasso_1se_Sunday <- glmnet(causality_x_Sunday, causality_y,alpha = 1, family = 'gaussian',lambda = causality_lam_1se_Sunday)
# Get the support
causality_supp1_Sunday <- support(causality_lasso_1se_Sunday$beta)
# Step 1 selected
length(causality_supp1_Sunday)
### controls
colnames(causality_x_Sunday[,causality_supp1_Sunday])

###
### Step 2.Select controls that are good to predict the treatment
causality_model2_Sunday <- cv.glmnet(causality_x_Sunday, treatment_Sunday, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_d_1se_Sunday <- causality_model2_Sunday$lambda.1se
# Call Lasso 
causality_lam_d_1se_Sunday <- glmnet(causality_x_Sunday, treatment_Sunday, alpha = 1, family = 'gaussian',lambda = causality_lam_d_1se_Sunday)
# Get the support
causality_supp2_Sunday <-support(causality_lam_d_1se_Sunday$beta)
### Step 2 selected
length(causality_supp2_Sunday)
### controls
colnames(causality_x_Sunday[,causality_supp2_Sunday])

###
### Step 3.Combine all selected and refit linear regression
causality_inthemodel_Sunday <- unique(c(causality_supp1_Sunday, causality_supp2_Sunday)) # unique grabs union
selectdata_Sunday <- cBind(treatment_Sunday, causality_x_Sunday[,causality_inthemodel_Sunday]) 
selectdata_Sunday <- as.data.frame(as.matrix(selectdata_Sunday)) # make it a data.frame
dim(selectdata_Sunday) ## p about half n

causality_lm_Sunday <- lm(causality_y ~ ., data = selectdata_Sunday)
summary(causality_lm_Sunday)$coef["treatment_Sunday",]

####
#### Compare/contrast the results with the following models
#### 
simple_Sunday <- lm(causality_y ~ weekdaySun, data = causality_data_train)
summary(simple_Sunday)
summary(full_fit)
summary(step_fit)
summary(lasso_fit)
summary(post_lasso)
coef(lower.tail)
coef(upper.tail)

### treatment is Business
causality_x_Business <- model.matrix(causality_y ~ .-channelBusiness, data = causality_data_train )
treatment_Business <- causality_data_train$channelBusiness

## Step 1.Select controls that are good to predict the outcome
causality_model1_Business <- cv.glmnet(causality_x_Business, causality_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_1se_Business <- causality_model1_Business$lambda.1se
# Call Lasso 
causality_lasso_1se_Business <- glmnet(causality_x_Business, causality_y,alpha = 1, family = 'gaussian',lambda = causality_lam_1se_Business)
# Get the support
causality_supp1_Business <- support(causality_lasso_1se_Business$beta)
# Step 1 selected
length(causality_supp1_Business)
### controls
colnames(causality_x_Business[,causality_supp1_Business])

###
### Step 2.Select controls that are good to predict the treatment
causality_model2_Business <- cv.glmnet(causality_x_Business, treatment_Business, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_d_1se_Business <- causality_model2_Business$lambda.1se
# Call Lasso 
causality_lam_d_1se_Business <- glmnet(causality_x_Business, treatment_Business, alpha = 1, family = 'gaussian',lambda = causality_lam_d_1se_Business)
# Get the support
causality_supp2_Business <-support(causality_lam_d_1se_Business$beta)
### Step 2 selected
length(causality_supp2_Business)
### controls
colnames(causality_x_Business[,causality_supp2_Business])

###
### Step 3.Combine all selected and refit linear regression
causality_inthemodel_Business <- unique(c(causality_supp1_Business, causality_supp2_Business)) # unique grabs union
selectdata_Business <- cBind(treatment_Business, causality_x_Business[,causality_inthemodel_Business]) 
selectdata_Business <- as.data.frame(as.matrix(selectdata_Business)) # make it a data.frame
dim(selectdata_Business) ## p about half n

causality_lm_Business <- lm(causality_y ~ ., data = selectdata_Business)
summary(causality_lm_Business)$coef["treatment_Business",]

####
#### Compare/contrast the results with the following models
#### 
simple_Business <- lm(causality_y ~ channelBusiness, data = causality_data_train)
summary(simple_Business)
summary(full_fit)
summary(step_fit)
summary(lasso_fit)
lasso_fit$terms
summary(post_lasso)
coef(lower.tail)
coef(upper.tail)

### treatment is World
causality_x_World <- model.matrix(causality_y ~ .-channelWorld, data = causality_data_train )
treatment_World <- causality_data_train$channelWorld

## Step 1.Select controls that are good to predict the outcome
causality_model1_World <- cv.glmnet(causality_x_World, causality_y, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_1se_World <- causality_model1_World$lambda.1se
# Call Lasso 
causality_lasso_1se_World <- glmnet(causality_x_World, causality_y,alpha = 1, family = 'gaussian',lambda = causality_lam_1se_World)
# Get the support
causality_supp1_World <- support(causality_lasso_1se_World$beta)
# Step 1 selected
length(causality_supp1_World)
### controls
colnames(causality_x_World[,causality_supp1_World])

###
### Step 2.Select controls that are good to predict the treatment
causality_model2_World <- cv.glmnet(causality_x_World, treatment_World, alpha = 1, family = 'gaussian', type.measure = 'mse')
causality_lam_d_1se_World <- causality_model2_World$lambda.1se
# Call Lasso 
causality_lam_d_1se_World <- glmnet(causality_x_World, treatment_World, alpha = 1, family = 'gaussian',lambda = causality_lam_d_1se_World)
# Get the support
causality_supp2_World <-support(causality_lam_d_1se_World$beta)
### Step 2 selected
length(causality_supp2_World)
### controls
colnames(causality_x_World[,causality_supp2_World])

###
### Step 3.Combine all selected and refit linear regression
causality_inthemodel_World <- unique(c(causality_supp1_World, causality_supp2_World)) # unique grabs union
selectdata_World <- cBind(treatment_World, causality_x_World[,causality_inthemodel_World]) 
selectdata_World <- as.data.frame(as.matrix(selectdata_World)) # make it a data.frame
dim(selectdata_World) ## p about half n

causality_lm_World <- lm(causality_y ~ ., data = selectdata_World)
summary(causality_lm_World)$coef["treatment_World",]

####
#### Compare/contrast the results with the following models
#### 
simple_World <- lm(causality_y ~ channelWorld, data = causality_data_train)
summary(simple_World)
summary(full_fit)
summary(step_fit)
summary(lasso_fit)
summary(post_lasso)
coef(lower.tail)
coef(upper.tail)