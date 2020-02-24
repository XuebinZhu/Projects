Mashable News Article Sharing Project
================
Danny Zhu
February 23, 2020

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax
for authoring HTML, PDF, and MS Word documents. For more details on
using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that
includes both content as well as the output of any embedded R code
chunks within the document. You can embed an R code chunk like this:

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for
example:

![](Increasing_News_Article_Sharing_on_Mashable_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the
plot.

``` r
                            #############################################
                            ### Mashable News Article Sharing Project ###
                            #############################################
rm(list=ls())

#load all the required library
library(factoextra)
```

    ## Warning: package 'factoextra' was built under R version 3.6.1

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.6.1

    ## Welcome! Related Books: `Practical Guide To Cluster Analysis in R` at https://goo.gl/13EFCZ

``` r
library(usmap)
```

    ## Warning: package 'usmap' was built under R version 3.6.1

``` r
library(randomForest)
```

    ## Warning: package 'randomForest' was built under R version 3.6.1

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
library(quantreg)
```

    ## Warning: package 'quantreg' was built under R version 3.6.1

    ## Loading required package: SparseM

    ## 
    ## Attaching package: 'SparseM'

    ## The following object is masked from 'package:base':
    ## 
    ##     backsolve

``` r
library(rpart)
```

    ## Warning: package 'rpart' was built under R version 3.6.1

``` r
library(rpart.plot)
```

    ## Warning: package 'rpart.plot' was built under R version 3.6.1

``` r
library(ggplot2)
library(tidyverse)
```

    ## Warning: package 'tidyverse' was built under R version 3.6.1

    ## -- Attaching packages --------------------------------------------------------------------------------------------------------- tidyverse 1.2.1 --

    ## v tibble  2.1.1       v purrr   0.3.2  
    ## v tidyr   0.8.3       v dplyr   0.8.0.1
    ## v readr   1.3.1       v stringr 1.4.0  
    ## v tibble  2.1.1       v forcats 0.4.0

    ## -- Conflicts ------------------------------------------------------------------------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::combine()       masks randomForest::combine()
    ## x dplyr::filter()        masks stats::filter()
    ## x dplyr::lag()           masks stats::lag()
    ## x randomForest::margin() masks ggplot2::margin()

``` r
library(dplyr)
library(MASS)
```

    ## Warning: package 'MASS' was built under R version 3.6.1

    ## 
    ## Attaching package: 'MASS'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     select

``` r
library(usmap)
library(glmnet)
```

    ## Warning: package 'glmnet' was built under R version 3.6.1

    ## Loading required package: Matrix

    ## Warning: package 'Matrix' was built under R version 3.6.1

    ## 
    ## Attaching package: 'Matrix'

    ## The following object is masked from 'package:tidyr':
    ## 
    ##     expand

    ## Loading required package: foreach

    ## Warning: package 'foreach' was built under R version 3.6.1

    ## 
    ## Attaching package: 'foreach'

    ## The following objects are masked from 'package:purrr':
    ## 
    ##     accumulate, when

    ## Loaded glmnet 2.0-18

``` r
library(MLmetrics)
```

    ## Warning: package 'MLmetrics' was built under R version 3.6.1

    ## 
    ## Attaching package: 'MLmetrics'

    ## The following object is masked from 'package:base':
    ## 
    ##     Recall

``` r
library(scales)
```

    ## Warning: package 'scales' was built under R version 3.6.1

    ## 
    ## Attaching package: 'scales'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     discard

    ## The following object is masked from 'package:readr':
    ## 
    ##     col_factor

``` r
library(plfm)
```

    ## Warning: package 'plfm' was built under R version 3.6.1

    ## Loading required package: sfsmisc

    ## Warning: package 'sfsmisc' was built under R version 3.6.1

    ## 
    ## Attaching package: 'sfsmisc'

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     last

    ## Loading required package: abind

``` r
library(caTools)
```

    ## Warning: package 'caTools' was built under R version 3.6.1

``` r
library(psych)
```

    ## Warning: package 'psych' was built under R version 3.6.1

    ## 
    ## Attaching package: 'psych'

    ## The following objects are masked from 'package:scales':
    ## 
    ##     alpha, rescale

    ## The following object is masked from 'package:MLmetrics':
    ## 
    ##     AUC

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     outlier

    ## The following objects are masked from 'package:ggplot2':
    ## 
    ##     %+%, alpha

``` r
library(VIM)
```

    ## Warning: package 'VIM' was built under R version 3.6.1

    ## Loading required package: colorspace

    ## Warning: package 'colorspace' was built under R version 3.6.1

    ## Loading required package: grid

    ## Loading required package: data.table

    ## Warning: package 'data.table' was built under R version 3.6.1

    ## 
    ## Attaching package: 'data.table'

    ## The following object is masked from 'package:sfsmisc':
    ## 
    ##     last

    ## The following objects are masked from 'package:dplyr':
    ## 
    ##     between, first, last

    ## The following object is masked from 'package:purrr':
    ## 
    ##     transpose

    ## VIM is ready to use. 
    ##  Since version 4.0.0 the GUI is in its own package VIMGUI.
    ## 
    ##           Please use the package to use the new (and old) GUI.

    ## Suggestions and bug-reports can be submitted at: https://github.com/alexkowa/VIM/issues

    ## 
    ## Attaching package: 'VIM'

    ## The following object is masked from 'package:datasets':
    ## 
    ##     sleep

``` r
#set seed for spliting data set
set.seed(123)
```

``` r
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
```

<img src="Increasing_News_Article_Sharing_on_Mashable_files/figure-gfm/unnamed-chunk-2-1.png" width="50%" /><img src="Increasing_News_Article_Sharing_on_Mashable_files/figure-gfm/unnamed-chunk-2-2.png" width="50%" />

``` r
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
```

    ##      Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
    ##    0.0000    1.0000    1.0000    0.9965    1.0000 1042.0000

``` r
boxplot(news$n_non_stop_words)
```

![](Increasing_News_Article_Sharing_on_Mashable_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
hist(news$n_non_stop_words)#this variable seems to have lots of errors, so I decided to remove it
```

![](Increasing_News_Article_Sharing_on_Mashable_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
#"is_weekend" is highly co-related with weekday, so I decided to remove it
#"timedelta" gives no predictive information for our y variable, so I decided to remove it
```
