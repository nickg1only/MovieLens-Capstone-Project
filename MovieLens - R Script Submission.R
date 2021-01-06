options(verbose = FALSE)
options(echo = FALSE)
options(warn = -1)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DATA PREPARATION:



# The following code will create the training and validation datasets:


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)


#Download the file containing the dataset:

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")


# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DATA CLEANING:



library(lubridate)


#Define cleaning function:

clean <- function(dataset){
  dataset %>% 
    mutate(timestamp = as.POSIXct(timestamp, origin = "1970-01-01")) %>%
    mutate(year = as.factor(year(timestamp)),
           month = as.factor(month(timestamp)))
}


#Clean the edx data:

edx_clean <- clean(edx)


#Clean the validation data:

validation_clean <- clean(validation)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DATA EXPLORATION:


cat("Training (edX) set:")
print(summary(edx_clean))

cat("Validation set:")
print(summary(validation_clean))



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DEFINING RMSE FUNCTION:



# This function calculates the RMSE given
#  the true ratings and the predicted ratings:

RMSE <- function(true_ratings, pred_ratings){
  sqrt(mean((true_ratings - pred_ratings)^2))
}



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#CREATING TEST & TRAINING DATASETS



# Create test and training datasets 
#  from the edx training dataset (NOT from the validation dataset):

test_index <- createDataPartition(y = edx$rating, 
                                  times = 1,
                                  p = 0.1, 
                                  list = FALSE)

train_set <- edx_clean[-test_index,]
test_set <- edx_clean[test_index,] %>%
  semi_join(train_set, by = "movieId") %>%  #This is to make sure the test set only has the same movies and users as the training set
  semi_join(train_set, by = "userId")


train_y <- train_set$rating

test_x <- test_set[,-3]
test_y <- test_set$rating



#Naive RMSE - the final model should do better:

mu_hat <- mean(train_y)
naive_rmse <- RMSE(test_y, mu_hat)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#DETERMINING BIASES:



#Calculate the movie-specific bias:

calc_movie_bias <- function(dataset){
  dataset %>%
    group_by(movieId) %>% 
    summarize(b_movie = mean(rating - mu_hat))
}


#Calculate the user-specific bias:

calc_user_bias <- function(dataset, movie_bias){
  dataset %>%
    left_join(movie_bias, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_user = mean(rating - mu_hat - b_movie))
}


#Calculate the genre-specific bias:

calc_genre_bias <- function(dataset, movie_bias, user_bias){
  dataset %>%
    left_join(movie_bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_genre = mean(rating - mu_hat - b_movie - b_user))
}


#Calculate the year-specific bias:

calc_year_bias <- function(dataset, movie_bias, user_bias, genre_bias){
  dataset %>%
    left_join(movie_bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    left_join(genre_bias, by = "genres") %>%
    group_by(year) %>%
    summarize(b_year = mean(rating - mu_hat - b_movie - b_user - b_genre))
}


#Calculate the month-specific bias:

calc_month_bias <- function(dataset, movie_bias, user_bias, genre_bias, year_bias){
  dataset %>%
    left_join(movie_bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    left_join(genre_bias, by = "genres") %>%
    left_join(year_bias, by = "year") %>%
    group_by(month) %>%
    summarize(b_month = mean(rating - mu_hat - b_movie - b_user - b_genre - b_year))
  
}



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#CREATE PREDICTIONS:



#Define a function to predict ratings given a dataset:

predict_ratings <- function(train, test){
  
  #Calculate the biases for the training set:
  
  movie_bias <- calc_movie_bias(train)
  user_bias <- calc_user_bias(train, movie_bias)
  genre_bias <- calc_genre_bias(train, movie_bias, user_bias)
  year_bias <- calc_year_bias(train, movie_bias, user_bias, genre_bias)
  month_bias <- calc_month_bias(train, movie_bias, user_bias, genre_bias, year_bias)
  
  
  #Use the biases to predict the ratings for the test set:
  
  pred_ratings <- test %>%
    left_join(movie_bias, by = "movieId") %>%
    left_join(user_bias, by = "userId") %>%
    left_join(genre_bias, by = "genres") %>%
    left_join(year_bias, by = "year") %>%
    left_join(month_bias, by = "month") %>%
    mutate(pred_rating = mu_hat + b_movie + b_user + b_genre + b_year + b_month) %>%
    pull(pred_rating)
  
  
  # Make sure the predicted ratings are 
  #  within the range 0.5 to 5:
  
  sapply(pred_ratings, function(rating){
    if(rating > 5){
      5
    }
    else{
      if(rating < 0.5){
        0.5
      }
      else{
        rating
      }
    }
  })
  
}



#Predict ratings for test set:

pred_test_ratings <- predict_ratings(train_set, test_x)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#TEST PREDICTIONS:



#Calculate RMSE for predictions on test set:

test_rmse <- RMSE(test_y, pred_test_ratings)

cat("This is the test RMSE, NOT the final RMSE: ", test_rmse)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#MAKE FINAL PREDICTIONS AND VALIDATE:



#Predict ratings on clean validation set using clean edx set:

final_pred_ratings <- predict_ratings(edx_clean, validation_clean[,-3])


#Calculate the final RMSE:

final_rmse <- RMSE(validation$rating, final_pred_ratings)


#Remove all other objects:

rm(edx_clean, test_index, test_set, test_x, train_set, mu_hat, naive_rmse, pred_test_ratings, test_rmse, test_y, train_y, calc_movie_bias, calc_user_bias, calc_genre_bias, calc_year_bias, calc_month_bias, clean, predict_ratings, RMSE)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#PRINT RESULTS:



cat("The final RMSE: ", final_rmse)
