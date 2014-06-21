
# code is commented because we downloaded the data once
# download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', destfile='pml-training.csv', method='curl');
# download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', destfile='pml-testing.csv', method='curl');

library(caret)

raw_training <- read.csv(".//data//pml-training.csv")
testing <- read.csv(".//data//pml-testing.csv")

set.seed(1)
inTrain <- createDataPartition(y = raw_training$classe, p = 0.6, list = F)
training <- raw_training[inTrain, ]
cv <- raw_training[-inTrain, ]

summary(training)

p1 <- qplot(new_window, num_window, data = training) + facet_grid(. ~ classe)
p1
p2 <- qplot(yaw_arm, total_accel_arm, data = training) + facet_grid(. ~ classe)
p2
p3 <- qplot(roll_arm, pitch_arm, data = training) + facet_grid(. ~ classe)
p3
p4 <- qplot(roll_belt, pitch_belt, data = training) + facet_grid(. ~ classe)
p4

input_vars_list <- c("new_window", "num_window", "roll_belt", "pitch_belt", 
    "yaw_belt", "total_accel_belt", "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", 
    "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "total_accel_dumbbell", 
    "roll_forearm", "pitch_forearm", "yaw_forearm", "total_accel_forearm")

exp_input <- function(x) {
    res = x[1]
    for (i in 2:length(x)) {
        res <- paste(res, " + ", x[i], sep = "")
    }
    return(res)
}

missClass <- function(values, prediction) {
    sum(prediction != values)/length(values)
}

input_vars <- exp_input(input_vars_list)


set.seed(2)

st <- Sys.time()
modFit <- train(eval(parse(text = paste("classe ~", input_vars, sep = ""))), 
    data = training, method = "rf")
	
et <- Sys.time()
et - st

missClass(training$classe, predict(modFit, training))

missClass(cv$classe, predict(modFit, cv))

answers <- predict(modFit, testing)
pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
pml_write_files(answers)	