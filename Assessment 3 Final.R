# Install required libraries
install.packages("ggplot2")
install.packages("tidyr")
install.packages("purrr")
install.packages("fastDummies")
install.packages("caret")
install.packages("DataExplorer")
install.packages("rpart.plot")
install.packages("rpart")
install.packages("ROCR")

# load libraries
library(Hmisc) # To plot histogram
library(ggplot2) # Visualization library
library(tidyr) # Data manipulation
library(purrr) # Handling functions and vectors
library(fastDummies) # Feature encoding function
library(caret) # Data Splitting & training model
library(DataExplorer) # Correlation matrix
library(rpart) # Decision Tree
library(rpart.plot) # Visualization for Decision Tree
library(ROCR) # ROC Curve

# Load data set
hotel_df <- read.csv("Hotel Reservations.csv", stringsAsFactors = TRUE, na.strings = c("", "NA"))

# *************** Basic Data Exploration ***************

# Print the first 6 rows of data frame
head(hotel_df) 

# Display the variable's names
names(hotel_df) 

# Display the list structure
str(hotel_df)

# Display the basic descriptive statistics
summary(hotel_df) 

# *************** Data Pre-processing **************** 

# ****************  Data Cleaning **************** 

# Remove irrelevant variables
hotel_df2 <- subset(hotel_df, select = -c(Booking_ID, arrival_year, arrival_date))

# Missing Values

# Display the number of missing (NULL/NA) values.
colSums(is.na(hotel_df2))

# Outliers

# Plot box plot to find outliers
hotel_df2 %>%
  keep(is.numeric) %>%                     # Keep only numeric columns
  gather() %>%                             # Convert to key-value pairs
  ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +     # In separate panels
  geom_boxplot()                           # Select box plot

# No outliers were deleted on "lead_time" and "avg_price_per_room" because customer usually pre-book their hotel room earlier and some of the room were free of charged.

# Correlation Matrix
plot_correlation(hotel_df2[, unlist(lapply(hotel_df2, is.numeric))])

# Remove one of the highly correlated variable
hotel_df2 <- subset(hotel_df2, select = -c(no_of_previous_bookings_not_canceled))

# ***************  Data Transformation **************** 

# ************ Standardization **************** 

# Robust scaling was selected because we would like to keep the outliers in the data

# Robust scaling function and calculation
robust_scalar<- function(x){(x- median(x)) /(quantile(x,probs = .75)-quantile(x,probs = .25))}

# Only "lead_time" and "avg_price_per_room" variables were selected to standardize as both are different scale to other numerical variables.
hotel_df3 <- hotel_df2   # Create new data set 

# Applying the robust_scalar function to both variables
hotel_df3[c("lead_time", "avg_price_per_room")] <- lapply(hotel_df2[c("lead_time", "avg_price_per_room")],robust_scalar)

# Rechecking the standardized values
summary(hotel_df3) 

# Rechecking the standardized values using density plot
hotel_df3 %>%
  keep(is.numeric) %>%                     # Keep only numeric columns
  gather() %>%                             # Convert to key-value pairs
  ggplot(aes(value)) +                     # Plot the values
  facet_wrap(~ key, scales = "free") +     # In separate panels
  geom_density()                           # Plot density plot

# **************** Feature Encoding **************** 

# "type_of_meal_plan", "room_type_reserved","market_segment_type", and "booking_status" were selected to undergo this process
summary(hotel_df3[c("type_of_meal_plan", "room_type_reserved","market_segment_type","booking_status")]) # to check how many levels were presented 

# One-code encoding
hotel_df3 <- dummy_columns(hotel_df3, select_columns = c('type_of_meal_plan','room_type_reserved'
                                                         ,'market_segment_type'), remove_first_dummy = TRUE) # Create dummy variables and remove the first dummy variable to avoid multicollinearity

# Remove original categorical variables from the data set
hotel_df3 <- subset(hotel_df3, select= -c(type_of_meal_plan,room_type_reserved,market_segment_type))

# Rechecking whether dummy variables were added
str(hotel_df3)

# **************** Data Splitting **************** 

set.seed(123) #random seed for reproducibility

split <- createDataPartition(hotel_df3$booking_status, p=0.7, list = FALSE) # Splitting into 70:30 ratio
training <- hotel_df3[split, ] # Passing 70% of the data into training data set
testing <- hotel_df3[-split, ] # Passing the remainder into testing data set

# **************** Data Imbalanced **************** 

barplot(prop.table(table(training$booking_status)), # To get the total count of unique value 
        col = rainbow(2),                           # Assign color to each bar
        ylim = c(0, 1),                             # Adjusting the height of y-axis 
        main = "Class Distribution")                # Title 

# **************** Predictive Modelling ****************

# **************** Logistic Regression (Cross Validation) ****************

# Resampling Technique

# Assigning method to resample 
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5, classProbs=TRUE)

set.seed(1111) # #random seed for reproducibility

# Train model using k-fold cross validation as control
model1 <- train(booking_status ~., data = training, 
                method = "glm", family = binomial, trControl = fitControl)

# Information of the model
print(model1)

# Residual & Coefficients
summary(model1)

# Predict testing data based on the model
prediction1 <- predict(model1, testing)
prediction1

# Create confusion matrix
confusionMatrix(prediction1, testing$booking_status)

# Features importance
varImp(model1)

# ROC-AUC curve
ROCPred1 <- prediction(as.numeric(prediction1), as.numeric(testing$booking_status)) 
ROCPer1 <- performance(ROCPred1, measure = "tpr", 
                       x.measure = "fpr")

# Plotting ROC Curve
plot(ROCPer1)
plot(ROCPer1, colorize = TRUE, 
     main = "ROC CURVE")
abline(a = 0, b = 1)

# AUC score
auc1 <- performance(ROCPred1, measure = "auc")
auc1 <- auc1@y.values[[1]]
auc1

# Rounding off the AUC score
auc1 <- round(auc1, 4)
# Insert AUC score into plot
legend(.6, .4, auc1, title = "AUC", cex = 1)


# **************** Decision Tree (Cross Validation) ****************

# Create another data set to avoid changing variables' names
hotel_df4 <- hotel_df3

# To make the variables' names into standard valid names for the rpart model
colnames(hotel_df4) <- make.names(colnames(hotel_df4))

# Data splitting for new data set
split2 <- createDataPartition(hotel_df4$booking_status, p=0.7, list = FALSE) # Splitting into 70:30 ratio
training2 <- hotel_df4[split, ] # Passing 70% of the data into training data set
testing2 <- hotel_df4[-split, ] # Passing the remainder into testing data set

# Training the decision tree model with cross validation
model2 <- train(booking_status ~., data = training2, 
                method = "rpart", trControl = fitControl, maxdepth = 20)

# Output of the trained decision tree model
summary(model2)

# Visualization of the decision tree
rpart.plot(model2$finalModel)

# Predict test data based on the trained model
prediction2 <- predict(model2, testing2)

# Create confusion matrix
confusionMatrix(prediction2, testing2$booking_status)

# Feature Importance
varImp(model2)

# ROC-AUC Curve
ROCPred2 <- prediction(as.numeric(prediction2), as.numeric(testing2$booking_status)) 
ROCPer2 <- performance(ROCPred2, measure = "tpr", 
                       x.measure = "fpr")

# AUC score
auc2 <- performance(ROCPred2, measure = "auc")
auc2 <- auc2@y.values[[1]]
auc2

# Plotting ROC Curve
plot(ROCPer2)
plot(ROCPer2, colorize = TRUE, 
     main = "ROC CURVE")
abline(a = 0, b = 1)

# Rounding off the AUC score
auc2 <- round(auc2, 4)
# Insert AUC score into the plot
legend(.6, .4, auc2, title = "AUC", cex = 1)
