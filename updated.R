rm(list = ls())

################################## Packages to install ##################################

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(knitr)) install.packages("knitr")
if(!require(rafalib)) install.packages("rafalib")


################################## Loading Data ##################################

ratings <- read.table(text = gsub("::", "\t", readLines(("ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(("ml-10M100K/movies.dat")), "\\::", 3)

# Giving column names to dataset and combining them together
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Seprating 10% of Movielens dataset for Validation set
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx  <- movielens[-test_index,]
temp <- movielens[test_index,]

# Making sure userId and MovieId of validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows back into edx set removed from valitaion set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Removing files no longer needed for project
rm(ratings, test_index, temp, movielens, removed)


################################## Exploratory analysis ##################################

# Dimenstion of edx set
dim(edx)

# Preview of edx set
head(edx)

# Checking for NA values
colSums(sapply(edx, is.na))

# Total number of Movies and User in edx set
edx %>% summarize(number_users = n_distinct(userId), number_movies = n_distinct(movieId))

# Rating scores and their count
edx %>% 
  group_by(rating) %>% 
  summarise(count = n()) %>% 
  arrange(desc(count)) %>% 
  kable()

# Review edx rating distribution
edx %>% 
  ggplot(aes(rating)) + 
  geom_histogram(binwidth=0.2, color="darkblue", fill="lightblue") + 
  ggtitle("Rating Distribution (Training")

# Table of user and movie
keep <- edx %>%
  dplyr::count(movieId) %>%
  top_n(5) %>%
  pull(movieId)

tab <- edx %>%
  filter(userId %in% c(20:40)) %>% 
  filter(movieId %in% keep) %>% 
  select(userId, title, rating) %>% 
  spread(title, rating)
tab %>% knitr::kable()

# Matrix of random 100 users and 100 where colour represent that we've rating for user & movie combination
users <- sample(unique(edx$userId), 100)

rafalib::mypar()
edx %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# Seperating year from title
edx <- edx %>% 
  mutate(releaseyear = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),
         title = str_remove(title, "[/(]\\d{4}[/)]$"))

# Number of movies per year/decade
movies_per_year <- edx %>%
  select(movieId, releaseyear) %>% # select columns we need
  group_by(releaseyear) %>% # group by year
  summarise(count = n())  %>% # count movies per year
  arrange(releaseyear)

movies_per_year %>%
  ggplot(aes(x = releaseyear, y = count)) +
  geom_line(color="blue")

# What were the most popular movie genres year by year?
genresByYear <- edx %>% 
  separate_rows(genres, sep = "\\|") %>% 
  select(movieId, releaseyear, genres) %>% 
  group_by(releaseyear, genres) %>% 
  summarise(count = n()) %>% arrange(desc(releaseyear))

# Different periods show certain genres being more popular during those periods
# It will be very hard to incorporate genre into overall prediction given this fact
ggplot(genresByYear, aes(x = releaseyear, y = count)) + 
  geom_col(aes(fill = genres), position = 'dodge') + 
  theme_hc() + 
  ylab('Number of Movies') + 
  ggtitle('Popularity per year by Genre')

# View release year vs rating
edx %>% group_by(releaseyear) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(releaseyear, rating)) +
  geom_point() +
  theme_hc() + 
  geom_smooth() +
  ggtitle("Release Year vs. Rating")


################################## Data Analysis ##################################

# Creating test and training set out of edx set for building and testing algorithm
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Making sure userId and MovieId of test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows back into edx set removed from valitaion set
removed <- anti_join(temp, test_set)
edx <- rbind(train_set, removed)

# Removing files which are not require anymore.
rm(temp, test_index, removed, genresByYear, movies_per_year)

# Loss funtion to measure the accuracy of our algorithm, in our case it'll be
# residual mean square error
RMSE <- function(actual_rating, predicted_rating){
  sqrt(mean((actual_rating - predicted_rating)^2))
}


################################## Recommendation algorithm ##################################

# Average of all ratings
mu_1 <- mean(train_set$rating)
mu_1

mean_rmse_1 <- RMSE(test_set$rating, mu_1)
mean_rmse_1

rmse_results <- data.frame(method = "Using average only", RMSE = mean_rmse_1)

# Average rating of individual movie(Movie effect)
mu_2 <- mean(train_set$rating)

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_2))

movie_avgs %>% 
  qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings <- mu_2 + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

mean_rmse_2 <- RMSE(predicted_ratings, test_set$rating)
mean_rmse_2

rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie Effect Model",
                                     RMSE = mean_rmse_2 ))
rmse_results %>% knitr::kable()

# Average rating of individual user(User effect)
train_set %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_2 - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_2 + b_i + b_u) %>%
  .$pred

mean_rmse_3 <- RMSE(predicted_ratings, test_set$rating)
mean_rmse_3

rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Movie + User Effects Model",  
                                     RMSE = mean_rmse_3 ))
rmse_results %>% knitr::kable()


################################## Tabels ##################################
# Pre-regularisation tabels

# We already see a big improvement. Can we make it better?
# Let's explore where we made mistakes. 
train_set %>% mutate(prediction = predicted_ratings, 
                residual   = predicted_ratings - temp$rating) %>%
  arrange(desc(abs(residual))) %>% 
  left_join(movies) %>%  
  select(title, prediction, residual) %>% slice(1:10)

qplot(b_i, geom = "histogram", 
      color = I("black"), fill=I("navy"), bins=25, data = movie_avgs)

# Let's create a database of movie title so that it's easy to show movie title otherwise it will show error
movie_titles <- train_set %>% 
  select(movieId, title) %>%
  distinct()

# These all seem like obscure movies. Many of them have large predictions
#Let's look at the top 10 worst and best movies
movie_avgs <-  left_join(movie_avgs, movies) 

# Here are the top ten movies:
arrange(movie_avgs, desc(b_i)) %>% 
  mutate(prediction = mu_2 + b_i) %>%
  select(title, prediction) %>% 
  slice(1:10)

# Here are the bottom ten:
arrange(movie_avgs, b_i) %>% 
  mutate(prediction = mu_2 + b_i) %>%
  select(title, prediction) %>% 
  slice(1:10)

# They all seem to be quite obscure. Let's look at how often they are rated.
edx %>%
  count(movieId) %>%
  left_join(movie_avgs) %>%
  arrange(desc(b_i)) %>% 
  mutate(prediction = mu_2 + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10)

edx %>%
  count(movieId) %>%
  left_join(movie_avgs) %>%
  arrange(b_i) %>% 
  mutate(prediction = mu_2 + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10)


################################## Penalty Term ##################################

#Let's find a value for penalty term which will be represented by lambda
#Lambda is tuning parameter ratings in our dataset so we can use cross validation to find the best penalty term
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  #movie average
  mu_1 <- mean(train_set$rating)
  
  #movie average
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_1)/(n()+l))
  
  #user average
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_1)/(n()+l))
  
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu_1 + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)

lambdas <- lambdas[which.min(rmses)]

rmse_results <- bind_rows(rmse_results,
                          data.frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()
################################## Tabels ##################################

# Post regularisation tabels
movie_reg_means <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu_1)/(n()+lambdas), n_i = n()) 

# Visualistion of how b_i has shrunken towards 0
data.frame(original = movie_avgs$b_i, 
           regularlized = movie_reg_means$b_i, 
           n = movie_reg_means$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Top 10 best after penalising
train_set %>%
  count(movieId) %>%
  left_join(movie_reg_means) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  mutate(prediction = mu_1 + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10)

# Top 10 worst after penalising
train_set %>%
  count(movieId) %>%
  left_join(movie_reg_means) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  mutate(prediction = mu_1 + b_i) %>%
  select(title, prediction, n) %>% 
  slice(1:10)