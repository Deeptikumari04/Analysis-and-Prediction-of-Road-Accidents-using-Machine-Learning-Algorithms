motor_crash =read.csv(("Motor_Vehicle_Crashes_-_Case_Information__Three_Year_Window.csv"), header=T, na.strings=c(""), stringsAsFactors = T)
motor_crash
motor_crash <- motor_crash[,-15]

#to check for null values
sapply(motor_crash,function(x) sum(is.na(x)))   #Municipality has 11977 missing data

#to add id column
motor_crash$id <- seq.int(nrow(motor_crash))

#to move id to 1st column
motor_crashNew <- motor_crash[,-18]
motor_crashNew
motor_crashNew <- cbind(motor_crash$id,motor_crashNew)
colnames(motor_crashNew)[1] <- "id"


#to split time in hh:mm to a new column in hh
motor_crashNew$hour <- format(strsplit(as.character(motor_crashNew$Time), "[:]"))
motor_crashNew$hour <- gsub("(.*),.*", "\\1", motor_crashNew$hour)
motor_crashNew

#to check if new created data frame has null value
sapply(motor_crashNew,function(x) sum(is.na(x)))

#to check format
str(motor_crashNew)

#getting rid of Pedestrian Bicyclist Action not required
motor_crashNew <- motor_crashNew[,-15]
motor_crashNew 


#getting rid of comma
motor_crashNew$Event.Descriptor <- lapply(motor_crashNew$Event.Descriptor, gsub, pattern=',',replacement='' )

#to check if any column value has comma in it
lapply(motor_crashNew, function(x) any(grepl(",", x)))
motor_crashNew
colnames(motor_crashNew)

write.csv(motor_crashNew, file ="motornew.csv",row.names=FALSE)

