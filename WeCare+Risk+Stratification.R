
####################################################
#######Assignment WeCare Risk Stratification########
#Read Data
#Evaluate Missing data and approach for Imputation
#Compute derived Metrics diabetes
#Removing unwanted Columns - Redundant Variables
#Changing readmitted to binary
#Convert Non factor columns
#Exploratory Data Analysis
#Prepare data for modelling
#Convert factor into binary variables
#Create dummy varibles
#Scale numeric attributes 
#Create train and test data
#Model Building
##Logistic Regression Model-1
##Model Evaluation {between Final Model and Step AIC Model}
####Apply KS-Statistics on Test Data
####Life and Gain Charts
##Random Forest Model-2
###Model evaluation
###Model Comparison
###Risk Stratification
####################################################

install.packages("randomForest","InformationValue","stringr","sqldf","ggplot2","dummies","naniar","afex","VIM","mice","dplyr","tidyr","car", dependencies = TRUE)
library(InformationValue)
library(ROCR)
library(dummies)
library(caret)
library(naniar)
library(mice)
library(VIM)
library(dplyr)
library(tidyr)
library(ggplot2)
library(MASS)
library(car)
library(stringr)
library(sqldf)
library(caTools)

#Read Data
setwd("C:/Masters/PGDDS/Health Care/WeCare Risk Stratification") #set working directory
diabetic_data <- read.csv("diabetic_data.csv",stringsAsFactors = TRUE) #load base data
str(diabetic_data) #data.frame':	101766 obs. of  50 variables. 
head(diabetic_data) #Confirmed that the Columns have question mark - ? Eg; Race, Weight, Payer Code, Diag 1/2/3, medical speciality

#Evaluate Missing data and approach for Imputation
sum(duplicated(diabetic_data$encounter_id)) #No duplicates on Encounter ID

ifelse(sapply(diabetic_data, function(x)all(is.na(x))) == TRUE, "Y","N") #No 'NA' Values across columns
#Based on Str data, checking columns for data issues
sqldf("Select race, count(encounter_id) as cnt from diabetic_data group by race") #Convert ? to Other
sqldf("Select gender, count(encounter_id) as cnt from diabetic_data group by gender") #3 records unknown/invalid, leave as is
sqldf("Select weight, count(encounter_id) as cnt from diabetic_data group by weight") #? found for 98569 records

colQuestionmarkCnt <- sqldf("select count(encounter_id) as cntID,
                            COUNT(CASE WHEN race == '?' then 1 ELSE NULL END) as raceCnt,
                            COUNT(CASE WHEN Weight == '?' then 1 ELSE NULL END) as WtCnt,
                            COUNT(CASE WHEN payer_code == '?' then 1 ELSE NULL END) as payerCnt,
                            COUNT(CASE WHEN medical_specialty == '?' then 1 ELSE NULL END) as msCnt,
                            COUNT(CASE WHEN diag_1 == '?' then 1 ELSE NULL END) as d1Cnt,
                            COUNT(CASE WHEN diag_2 == '?' then 1 ELSE NULL END) as d2Cnt,
                            COUNT(CASE WHEN diag_3 == '?' then 1 ELSE NULL END) as d3Cnt
                            from diabetic_data")


na_strings <- c("?") #Convert ? to NA Values
diabetic_data %>% replace_with_na(condition = ~.x %in% na_strings) #Convert ? to NA Values
pMiss <- function(x){sum(is.na(x))/length(x)*100} #Convert ? to NA Values
apply(diabetic_data,2,pMiss) #apply(diabetic_data,1,pMiss)
#Inferences from Missing Data - MNAR {Missing Not At Random scenario} - A more serious issue and in this case it might be wise to check the data gathering process further and try to understand why the information is missing

#Diagnosis 1/2/3 are Diagnosis ID. Before applying MICE we can identify if a Patient is diabetic or not based on dependent columns of diabetesMed and A1Cresult. 
#Current NA values are diag1 - 21 observations - 0.02% of rows ; diag2 - 358 observations - 0.35% ; diag3 - 1423 observations - 1.40%
diabetic_data$diag_1 <- ifelse((is.na(diabetic_data$diag_1) & diabetic_data$diabetesMed =="Yes") | (is.na(diabetic_data$diag_1)  & diabetic_data$A1Cresult  !="None"),
                               as.numeric(250),
                               diabetic_data$diag_1) 

diabetic_data$diag_2 <- ifelse((is.na(diabetic_data$diag_2) & diabetic_data$diabetesMed =="Yes") | (is.na(diabetic_data$diag_2)  & diabetic_data$A1Cresult  !="None"),
                               as.numeric(250),
                               diabetic_data$diag_2) 

diabetic_data$diag_3 <- ifelse((is.na(diabetic_data$diag_3) & diabetic_data$diabetesMed =="Yes") | (is.na(diabetic_data$diag_3)  & diabetic_data$A1Cresult  !="None"),
                               as.numeric(250),
                               diabetic_data$diag_3) 

pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(diabetic_data,2,pMiss)

#Missing Values reduced to diag1 - 0.005%, diag2 - 0.043 and diag3% - 0.24% 
#Weight has 98569 observations - 96.86%, Payer has 40256 observations - 39.56%, Medical Speciality has 49949 observations - 49.08%% 
#Using mice for looking at missing data pattern - https://www.r-bloggers.com/imputing-missing-data-with-r-mice-package/
missingValesData <- data.frame("encounterID" = diabetic_data$encounter_id,
                               "diag1"  = diabetic_data$diag_1,
                               "diag2"  = diabetic_data$diag_2,
                               "diag3"  = diabetic_data$diag_3,
                               "wt"  = diabetic_data$weight,
                               "Payr"  = diabetic_data$payer_code,
                               "medSpe" = diabetic_data$medical_specialty)

md.pattern(missingValesData)
aggr_plot <- aggr(missingValesData, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(missingValesData), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
sum(is.na(missingValesData$diag1))
#Tying predictive mean matching as imputation method.
tempData <- mice(missingValesData,m=7,maxit=50,meth='pmm',seed=500)
summary(tempData)
#No need for mice. This data set is completely observed. Evaluation of PMM method - Matrix is not invertible due to collinearity and determinant equals zero. Hence we manually fix 
#Manually fix diag1/2/3, For the benefit of Risk Stratification, we assume the missing values are Diabetic, so they come into the model and can be evaluated based on model
#We do not need Weight, Medical Speciality and Payer for Risk Stratification. Hence we drop these

#Reload Data 
rm("diabetic_data")
diabetic_data <- read.csv("diabetic_data.csv",stringsAsFactors = TRUE) #load base data
str(diabetic_data) #data.frame':	101766 obs. of  50 variables. 

#Based on mice() - predictive mean matching as imputation method, we determined to consider all ? to 250
#Compute derived Metrics diabetes, circulatoryDisease and comorbidity
#If diabetic then 1, else 0
diabetic_data$diabetic <- ifelse(diabetic_data$diag_1=="?",
                                 as.numeric(1),
                                 as.numeric(0))

diabetic_data$diabetic <- ifelse(diabetic_data$diag_2=="?",
                                 as.numeric(1),
                                 as.numeric(0))

diabetic_data$diabetic <- ifelse(diabetic_data$diag_3=="?",
                                 as.numeric(1),
                                 as.numeric(0))

diabetic_data$diabetic <- ifelse(grepl("250", diabetic_data$diag_1) | grepl("250", diabetic_data$diag_2) | grepl("250", diabetic_data$diag_3),
                                 as.numeric(1),
                                 as.numeric(0)) 

sqldf("Select diabetic,count(*) from diabetic_data group by diabetic") #38024 diabetic patients where 1423 are patients with missing values considered as diabetic
#If Circulatory disease 1, else 0
diabetic_data$diag_1 <- as.numeric(as.character(diabetic_data$diag_1))
diabetic_data$diag_2 <- as.numeric(as.character(diabetic_data$diag_2))
diabetic_data$diag_3 <- as.numeric(as.character(diabetic_data$diag_3))
diabetic_data$circulatoryDisease <- ifelse(as.numeric(diabetic_data$diag_1) >= "390" & as.numeric(diabetic_data$diag_1) <= "459" | as.numeric(diabetic_data$diag_2) >= "390" & as.numeric(diabetic_data$diag_2) <= "459" | as.numeric(diabetic_data$diag_3) >= "390" & as.numeric(diabetic_data$diag_3) <= "459",
                                           as.numeric(1),
                                           as.numeric(0)) 
sqldf("Select circulatoryDisease, count(circulatoryDisease) cnt from diabetic_data group by circulatoryDisease")
#Only 1 and 0 values are present, 20466 rows with derived metric

#Comorbidity_Value based on Matrix of Diabetes (250.xx) - Circulatory Disease (390-459)
diabetic_data$comorbidity <- ifelse(as.numeric(diabetic_data$circulatoryDisease + diabetic_data$diabetic) == 2,
                                    3,
                                    ifelse(diabetic_data$circulatoryDisease == 0 & diabetic_data$diabetic == 1,
                                           1,
                                            ifelse(diabetic_data$circulatoryDisease == 1 & diabetic_data$diabetic == 0,
                                                  2,
                                                  0))) 

unique(diabetic_data$comorbidity) #6052 NA values which are non-diabetic and non-circulatoryDisease hence changing to 0. These are V* codes
diabetic_data$comorbidity <- ifelse(is.na(diabetic_data$comorbidity),
                                 0,
                                 diabetic_data$comorbidity)
sqldf("Select comorbidity, count(encounter_id) as c1 from diabetic_data group by comorbidity ") #Has only 0, 1, 2 and 3 Values
# comorbidity    count
#1           0 25462
#2           1 16565
#3           2 41248
#4           3 18491

#Race has 2273 observations - 2.23% of rows --> We convert Race of 2273 as Other as it is only 2% of data
diabetic_data$race <- ifelse(diabetic_data$race=="?",
                             as.character("Other"),
                             as.character(diabetic_data$race)) 
levels(factor(diabetic_data$race))
unique(diabetic_data$race)

#Removing unwanted Columns - Redundant Variables
colsdontwant <- c("weight","payer_code","medical_specialty", "metformin",	"repaglinide",	"nateglinide",	"chlorpropamide",	"glimepiride",	"acetohexamide",	"glipizide",	"glyburide",	"tolbutamide",	"pioglitazone",	"rosiglitazone",	"acarbose",	"miglitol",	"troglitazone",	"tolazamide",	"examide",	"citoglipton",	"glyburide-metformin",	"glipizide-metformin",	"glimepiride-pioglitazone",	"metformin-rosiglitazone",	"metformin-pioglitazone", "glyburide.metformin","glipizide.metformin","glimepiride.pioglitazone","metformin.rosiglitazone","metformin.pioglitazone","max_glu_serum")
diabetic_data <- diabetic_data[ , !(names(diabetic_data) %in% colsdontwant), drop = FALSE]
str(diabetic_data)

#Changing readmitted to binary
diabetic_data$mod_readmitted <- ifelse(diabetic_data$readmitted =="<30" | diabetic_data$readmitted ==">30",
                                       as.character("YES"),
                                       as.character("NO")) 
sqldf("Select distinct mod_readmitted from diabetic_data") #Only YES and NO values are present

#Removing diagnosis columns as these have been used to calculate Combordity
colsdontwant <- c("diag_1","diag_2","diag_3")
diabetic_data <- diabetic_data[ , !(names(diabetic_data) %in% colsdontwant), drop = FALSE]

#Convert Non factor columns which are required to obtain levels into Factors 
str(diabetic_data)
diabetic_data$diabetic <- as.factor(diabetic_data$diabetic)
diabetic_data$circulatoryDisease <- as.factor(diabetic_data$circulatoryDisease)
diabetic_data$comorbidity <- as.factor(diabetic_data$comorbidity)
diabetic_data$mod_readmitted <- as.factor(diabetic_data$mod_readmitted)
diabetic_data$race <- as.factor(diabetic_data$race)
diabetic_data$admission_type_id <- as.factor(diabetic_data$admission_type_id)
diabetic_data$discharge_disposition_id <- as.factor(diabetic_data$discharge_disposition_id)
diabetic_data$admission_source_id <- as.factor(diabetic_data$admission_source_id)
diabetic_data$time_in_hospital <- as.numeric(diabetic_data$time_in_hospital)
#ggsave("cbg.pdf",plot = last_plot(),path="C:/Masters/PGDDS/Health Care/WeCare Risk Stratification")


####Begin Exploratory Data Analysis
#1. Missing Data - MNAR {Missing Not At Random scenario} - A more serious issue and in this case it might be wise to check the data gathering process further and try to understand why the information is missing
#G1. Comorbidity 2 is higher in male and female
ggplot(diabetic_data,aes(x=comorbidity,fill= gender))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("Comorbidity") + ylab("Gender") + ggtitle("Comorbidity by Gender ") 
#G2. Female population is higher and Caucasian, African are top 2 race
ggplot(diabetic_data,aes(x=race,fill= gender))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("race") + ylab("Gender") + ggtitle("race & Gender ") 
#G3. Patients with Circulatory diseases seems to be more than diabetic. High(2) and Very High (3)comorbidity is dominant from age >= 40 
ggplot(diabetic_data,aes(x=age,fill= comorbidity))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("age") + ylab("comorbidity") + ggtitle("Age & comorbidity ") 
#G4 Health Care seems to be working well for more than 50% of patients as they are not re-admitted
ggplot(diabetic_data,aes(x=age,fill= readmitted))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("age") + ylab("readmitted") + ggtitle("Age & readmitted ") 
#G5 ALOC has a downward trend with low in young aged group (<40), more in middle aged group (40-60), high above >60  
ggplot(diabetic_data,aes(x=time_in_hospital,fill= age))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("time_in_hospital") + ylab("age") + ggtitle("Age & ALOC") 
#G6 Circulatory Disease (390-459) has more lab procedures
ggplot(diabetic_data,aes(x=num_lab_procedures,fill=comorbidity ))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("num_lab_procedures") + ylab("comorbidity") + ggtitle("num_lab_procedures & comorbidity") 
#G7 Number of medications is higher in circulatory disease than diabetic
ggplot(diabetic_data,aes(x=num_medications,fill=comorbidity ))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("num_medications") + ylab("comorbidity") + ggtitle("num_medications & comorbidity") 
ggplot(diabetic_data,aes(x=num_medications,fill=diabetic ))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("num_medications") + ylab("diabetic") + ggtitle("num_medications & diabetic") 
ggplot(diabetic_data,aes(x=num_medications,fill=circulatoryDisease ))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("num_medications") + ylab("circulatoryDisease") + ggtitle("num_medications & circulatoryDisease") 
#G8 Inpatients are more for comorbidity 0 & 2
ggplot(diabetic_data,aes(x=age,fill=comorbidity))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("age") + ylab("number_inpatient") + ggtitle("In Patients & comorbidity") 
#G9 Outpatients are more for comorbidity 2 and 3
ggplot(diabetic_data,aes(x=age,fill=comorbidity))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("age") + ylab("number_outpatient") + ggtitle("Out Patients & comorbidity") 
#G10 Emergency cases are very high in middle/old aged patients with almost equal split of comorbidity
ggplot(diabetic_data,aes(x=age,fill=comorbidity))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("age") + ylab("number_emergency") + ggtitle("Emergency & comorbidity") 
#G11 Insulin is prescribed to patients without diabetes. Some diabetic patients do not have insulin prescribed
ggplot(diabetic_data,aes(x=insulin,fill=comorbidity))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("insulin") + ylab("comorbidity") + ggtitle("Insulin & comorbidity") 
#G12 diabetes med is prescribed to non-diabetic patients because of A1Cresult
ggplot(diabetic_data,aes(x=diabetesMed,fill=A1Cresult))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("diabetesMed") + ylab("comorbidity") + ggtitle("comorbidity, diabetesMed and A1Cresult") 
################################################End of EDA

####Prepare data for modelling
#From lecture following are important- Time in hospital, Number of lab procedures, Number of medications, Number of outpatient visits, Number of emergency visits, Number of inpatient visits, Readmitted, Comorbidity, Change of medications, A1c test result
#From EDA, we observed that Gender, Race and Age are important
reqColNames <- c("number_emergency","number_inpatient","mod_readmitted","comorbidity","change","gender","time_in_hospital","num_lab_procedures","num_medications","number_outpatient")
modellingData <- diabetic_data[ , reqColNames]
colnames(diabetic_data) 
reqColNames <- c("race","age","A1Cresult","insulin")
chrmodellingData <- diabetic_data[ , reqColNames]
str(modellingData)

#Convert factor into binary variables - mod_readmitted, change,gender
modellingData$mod_readmitted <- ifelse(modellingData$mod_readmitted =="YES",1,0) 
modellingData$change <- ifelse(modellingData$change =="Yes",0,1)
modellingData$gender <- ifelse(modellingData$gender =="Female" | modellingData$gender =="Unknown/Invalid",
                                           1,
                                           0) 

str(modellingData)
#Create dummy varibles for categorical ones - insulin, A1cresult, race, age,
fmodellingdata<- data.frame(sapply(chrmodellingData, function(x) factor(x)))
dummies_fact<- data.frame(sapply(fmodellingdata , 
                                 function(x) data.frame(model.matrix(~x-1,data =fmodellingdata))[,-1]))

modellingData<- cbind(modellingData,dummies_fact) 
#Delete  c("race","age","A1Cresult","insulin") fields
colnames(modellingData)
reqColNames <- c("race","age","A1Cresult","insulin")
modellingData <- modellingData[ , !(names(modellingData) %in% reqColNames), drop = FALSE]
str(modellingData)

#Scale numeric attributes 
modellingData$number_outpatient <- scale(modellingData$number_outpatient,center = TRUE, scale = TRUE)
modellingData$num_medications <- scale(modellingData$num_medications,center = TRUE, scale = TRUE)
modellingData$num_lab_procedures <- scale(modellingData$num_lab_procedures,center = TRUE, scale = TRUE)
modellingData$time_in_hospital <- scale(modellingData$time_in_hospital,center = TRUE, scale = TRUE)
modellingData$number_inpatient <- scale(modellingData$number_inpatient,center = TRUE, scale = TRUE)
modellingData$number_emergency <- scale(modellingData$number_emergency,center = TRUE, scale = TRUE)

#creating test and train data
set.seed(100) 
trainIndex = sample.split(modellingData$mod_readmitted, SplitRatio = 0.7)
diabetes_Train = modellingData[trainIndex,]
diabetes_Test = modellingData[!(trainIndex),]
str(diabetes_Train)

##############################Model Building##################################################
#########################Logistic Regression Model-1#########################################
#########################risk of readmission for the patient

firstModel = glm(mod_readmitted ~ ., data = diabetes_Train, family = "binomial")
summary(firstModel) #AIC: 78515, Residual deviance 71209

stepAICModel<- stepAIC(firstModel, direction="both")
summary(stepAICModel)
vif(stepAICModel)
#Suggested Model
#glm(formula = mod_readmitted ~ number_emergency + number_inpatient + 
#comorbidity + gender + time_in_hospital + num_lab_procedures + 
#  num_medications + number_outpatient + race.xAsian + race.xCaucasian + 
#  race.xHispanic + race.xOther + age.x.10.20. + age.x.20.30. + 
#  age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
#  age.x.70.80. + age.x.80.90. + age.x.90.100. + A1Cresult.xNorm + 
#  insulin.xNo + insulin.xSteady + insulin.xUp, family = "binomial", 
#data = diabetes_Train)

#remove hispanic based on pvalue and significance
thirdModel = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                   comorbidity + gender + time_in_hospital + num_lab_procedures + 
                   num_medications + number_outpatient + race.xAsian + race.xCaucasian + 
                    race.xOther + age.x.10.20. + age.x.20.30. + 
                   age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                   age.x.70.80. + age.x.80.90. + age.x.90.100. + A1Cresult.xNorm + 
                   insulin.xNo + insulin.xSteady + insulin.xUp, family = "binomial", 
                 data = diabetes_Train)
summary(thirdModel)
vif(thirdModel)
#Slight increase in VIF

#remove insulin.xUp based on high pvalue
model_4 = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                   comorbidity + gender + time_in_hospital + num_lab_procedures + 
                   num_medications + number_outpatient + race.xAsian + race.xCaucasian + 
                   race.xOther + age.x.10.20. + age.x.20.30. + 
                   age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                   age.x.70.80. + age.x.80.90. + age.x.90.100. + A1Cresult.xNorm + 
                   insulin.xNo + insulin.xSteady, family = "binomial", 
                 data = diabetes_Train)
summary(model_4)
vif(model_4) #No change i n VIF

#remove num_medications based on low significance and pvalue
model_5 = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                comorbidity + gender + time_in_hospital + num_lab_procedures + 
                number_outpatient + race.xAsian + race.xCaucasian + 
                race.xOther + age.x.10.20. + age.x.20.30. + 
                age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                age.x.70.80. + age.x.80.90. + age.x.90.100. + A1Cresult.xNorm + 
                insulin.xNo + insulin.xSteady, family = "binomial", 
              data = diabetes_Train)
summary(model_5)
vif(model_5) #slight increase in VIF

#remove age.x.10.20 based on low significance and pvalue
model_6 = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                comorbidity + gender + time_in_hospital + num_lab_procedures + 
                number_outpatient + race.xAsian + race.xCaucasian + 
                race.xOther + age.x.20.30. + 
                age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                age.x.70.80. + age.x.80.90. + age.x.90.100. + A1Cresult.xNorm + 
                insulin.xNo + insulin.xSteady, family = "binomial", 
              data = diabetes_Train)
summary(model_6)
vif(model_6) #increase in VIF

#remove age.x.20.30. based on low significance and pvalue
model_7 = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                comorbidity + gender + time_in_hospital + num_lab_procedures + 
                number_outpatient + race.xAsian + race.xCaucasian + 
                race.xOther +  
                age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                age.x.70.80. + age.x.80.90. + age.x.90.100. + A1Cresult.xNorm + 
                insulin.xNo + insulin.xSteady, family = "binomial", 
              data = diabetes_Train)
summary(model_7)
vif(model_7) #decrease in VIF

#remove age.x.90.100 based on low significance and high pvalue
model_8 = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                comorbidity + gender + time_in_hospital + num_lab_procedures + 
                number_outpatient + race.xAsian + race.xCaucasian + 
                race.xOther +  
                age.x.30.40. + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                age.x.70.80. + age.x.80.90. + A1Cresult.xNorm + 
                insulin.xNo + insulin.xSteady, family = "binomial", 
              data = diabetes_Train)
summary(model_8)
vif(model_8) #decrease in VIF

#remove age.x.30.40 basedon low significance and high pvalue
model_9 = glm(mod_readmitted ~ number_emergency + number_inpatient + 
                comorbidity + gender + time_in_hospital + num_lab_procedures + 
                number_outpatient + race.xAsian + race.xCaucasian + 
                race.xOther + age.x.40.50. + age.x.50.60. + age.x.60.70. + 
                age.x.70.80. + age.x.80.90. + A1Cresult.xNorm + 
                insulin.xNo + insulin.xSteady, family = "binomial", 
              data = diabetes_Train)
summary(model_9)
vif(model_9)
#GVIF is normal, VIF has slightly increased , all Pvalues are low
#cannot further trim the model. Model 9 is final model


finalModel <- model_9 #Cannot trim the model further as all variables are significant. Eigth Model is final model

############################Model Evaluation
#Model evaluation on StepAIC Model, Final Model and seventhModel
stepAICModelTestPrediction <- predict(stepAICModel, type="response", newdata = dplyr::select(diabetes_Test, -mod_readmitted))
finalmodelTestPrediction <- predict(finalModel, type="response", newdata = dplyr::select(diabetes_Test, -mod_readmitted))
finalModeldiabetesTest <- diabetes_Test
finalModeldiabetesTest$finalProbability <- finalmodelTestPrediction
summary(finalModeldiabetesTest)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.20    0.38    0.43    0.46    0.51    1.00

stepAICModeldiabetesTest <-diabetes_Test
stepAICModeldiabetesTest$stepAICProbability <- stepAICModelTestPrediction
summary(stepAICModelTestPrediction)
#Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#0.13    0.38    0.42    0.46    0.51    1.00 

#Probababilitycutoff - 50%.
finalModelPrediction <- factor(ifelse(finalmodelTestPrediction >= 0.50, "Yes", "No"))
finalModelActual <- factor(ifelse(finalModeldiabetesTest$mod_readmitted==1,"Yes","No"))
table(finalModelPrediction , finalModelActual )
#                      finalModelActual
#finalModelPrediction    No   Yes
#                  No  13440  8776
#                  Yes  3019  5295

stepAICModelPrediction <- factor(ifelse(stepAICModelTestPrediction >= 0.50, "Yes", "No"))
stepAICModelActual <- factor(ifelse(stepAICModeldiabetesTest$mod_readmitted==1,"Yes","No"))
table(stepAICModelPrediction , stepAICModelActual )
#                        stepAICModelActual
#stepAICModelPrediction    No   Yes
#                    No  13501  8844
#                    Yes  2958  5227

Risk_Stratification_Test_conf <- confusionMatrix(finalModelPrediction, finalModelActual, positive = "Yes")
finalModelTestconfusionMatx <- confusionMatrix(finalModelPrediction, finalModelActual, positive = "Yes")
stepAICTestconfusionMatx <- confusionMatrix(stepAICModelPrediction, stepAICModelActual, positive = "Yes")
#Accuracy of 0.614, Sensitivity : 0.376, Specificity : 0.817, Balanced Accuracy : 0.596

finalModelTestconfusionMatx
#Accuracy of 0.613, Sensitivity : 0.371, Specificity : 0.820 , Balanced Accuracy : 0.596 

stepAICTestconfusionMatx
#Not much of a difference between Final Model and Step AIC, hence dropping off AIC as Final model is better

#Finding Optimal Probability Cutoff
summary(finalmodelTestPrediction)
min(finalModeldiabetesTest$finalProbability) #0.2
max(finalModeldiabetesTest$finalProbability) #1
library(tidyverse)

perform_fn <- function(cutoff) 
{
  predicted_att <- factor(ifelse(finalmodelTestPrediction >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_att, finalModelActual, positive = "Yes")
  acc <- conf$overall[1]
  sens <- conf$byClass[1]
  spec <- conf$byClass[2]
  out <- t(as.matrix(c(sens, spec, acc))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}
# Creating cutoff values - finalModel 0.2 to 1 for plotting and initiallizing a matrix of 100 X 3.

s = seq(.01,.80,length=100)
OUT = matrix(0,100,3)

for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 

plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend(0,.50,col=c(2,"darkgreen",4,"darkred"),lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))

cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.05)]
#0.42 sensitivity and specificity are mutually near
testCutOffAttribute <- factor(ifelse(finalmodelTestPrediction >=0.42, "Yes", "No"))
finalConfusionMatrix <- confusionMatrix(testCutOffAttribute, finalModelActual, positive = "Yes")

modelaccuracy <- finalConfusionMatrix$overall[1]
#0.6
modelSensitivity <- finalConfusionMatrix$byClass[1]
#0.64
modelspecificity <- finalConfusionMatrix$byClass[2]
#0.57


###Apply KS-Statistics on Test Data
testCutOffAttribute <- ifelse(testCutOffAttribute == "Yes", 1, 0)
finalModelActual <- ifelse(finalModelActual == "Yes", 1, 0)
predictionObjectTest<- prediction(testCutOffAttribute, finalModelActual)
performanceMeasuresTest<- performance(predictionObjectTest, "tpr", "fpr")
ksTableTest <- attr(performanceMeasuresTest, "y.values")[[1]] - (attr(performanceMeasuresTest, "x.values")[[1]])
max(ks_table_test) #0.21, not more than 40%

###Life and Gain Charts
lift <- function(labels , predicted_prob,groups=10) {
  
  if(is.factor(labels)) labels  <- as.integer(as.character(labels ))
  if(is.factor(predicted_prob)) predicted_prob <- as.integer(as.character(predicted_prob))
  helper = data.frame(cbind(labels , predicted_prob))
  helper[,"bucket"] = ntile(-helper[,"predicted_prob"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(labels ), funs(total = n(),
                                     totalresp=sum(., na.rm = TRUE))) %>%
    
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups))) 
  return(gaintable)
}
readmittedPatientsDecile = lift(finalModelActual, finalmodelTestPrediction, groups = 10)
#Evident that the totalresp is in top 4 deciles
#bucket total totalresp Cumresp  Gain Cumlift
#<int> <int>     <dbl>   <dbl> <dbl>   <dbl>
#1      1  3053      2234    2234  15.9    1.59
#2      2  3053      1857    4091  29.1    1.45
#3      3  3053      1631    5722  40.7    1.36
#4      4  3053      1536    7258  51.6    1.29
#5      5  3053      1409    8667  61.6    1.23
#6      6  3053      1284    9951  70.7    1.18
#7      7  3053      1163   11114  79.0    1.13
#8      8  3053      1123   12237  87.0    1.09
#9      9  3053      1016   13253  94.2    1.05
#10     10  3053       818   14071 100      1   

#Plotting Gain and Lift Charts
ks_plot(finalModelActual, testCutOffAttribute)
plot(readmittedPatientsDecile$Cumlift, type="l", lwd=2, col="red",
     xlim = c(0,10),
     ylim = c(0,4),
     main = "Lift Chart",
     xlab = "Decile",
     ylab = "Lift")
abline(h=1, col="brown")
axis(1, 1:10)
abline(h=0:10, v=0:10, lty=3)

#########################Random Forest Model-2#########################################

library(randomForest)
set.seed(100) 
trainIndexRF = sample.split(modellingData$mod_readmitted, SplitRatio = 0.80)
diabetes_TrainRF = modellingData[trainIndex,]
diabetes_TestRF = modellingData[!(trainIndex),]
modelRandomForest <- randomForest(mod_readmitted ~ ., 
                                  data=diabetes_TrainRF, 
                                  importance=TRUE,
                                  na.action=na.omit,
                                  ntree=50, 
                                  mtry=5, 
                                  do.trace=TRUE,
                                  proximity=FALSE)
summary(modelRandomForest)
#feature importance plot
varImpPlot(modelRandomForest)
#Mean of squared residuals: 0.2334802
#% Var explained: 6.3

#Model evaluation
modelRandomForestPredictions <- predict(modelRandomForest, na.action = na.pass, type="response", newdata = dplyr::select(diabetes_TestRF, -mod_readmitted))
diabetes_TestRF$mod_readmitted_PredictedRF <- modelRandomForestPredictions
predictedProbabilityRF <- c(max(diabetes_TestRF$mod_readmitted_PredictedRF),min(diabetes_TestRF$mod_readmitted_PredictedRF ))
#0.99124519 to 0.04362732

#Applying probability cutoff of 40%
rfTestPredictionAtt <- factor(ifelse(modelRandomForestPredictions >= 0.55, "Yes", "No"))
rfActualPredictionAtt <- factor(ifelse(diabetes_TestRF$mod_readmitted == 1,"Yes","No"))
table(rfTestPredictionAtt, rfActualPredictionAtt)
#                      rfActualPredictionAtt
#rfTestPredictionAtt    No   Yes
#                 No   4812  2078
#                 Yes 11647 11993
library(caret)
cmRF <- confusionMatrix(rfTestPredictionAtt, rfActualPredictionAtt)
#Accuracy of 0.6114
#Specificity : 0.3620
#Sensitivity : 0.8246
##########################End of Random Forest###################

#########################Model Comparison#######################
#Logistic Regression Model 1 has accuracy of 60% while Random Forest has 61%
#Specificity is high in Model 1 than RF
#Sensitivity is low in MOdel 1 than RF
#Given the time taken for RF in execution and comparitively higher in 2 parameters, 
#Random Forest is preferred model for Risk Stratification
#########################Risk Stratification#########################

#Stratify your population into 3 risk buckets: Using Radom Forest Model
modellingData1 <- modellingData
modellingData$readmitPredicted <- predict(modelRandomForest, newdata=modellingData)
modellingData$RiskStratificationPopulation <- ifelse(modellingData$readmitPredicted >= 0.7,
                                              "High Risk",
                                              ifelse(modellingData$readmitPredicted > 0.3,
                                              "Medium Risk",  
                                              "Low Risk"))
  
modellingData$RiskStratificationPopulation <- as.factor(modellingData$RiskStratificationPopulation)
ggplot(modellingData,aes(x=RiskStratificationPopulation,fill=comorbidity))+geom_bar(position = "stack")+scale_y_continuous(trans='log2') +geom_text(size = 3, position = position_stack(vjust = 0.5),stat='count',aes(label=..count..),vjust=2)+ xlab("RiskStratificationPopulation") + ylab("comorbidity") + ggtitle("Risk Stratification - Probability of readmission") 
#12357 High Risk, 18251 Medium Risk, 71158 Low Risk 
#Risk Stratification	Comorbidity			
#                  0	    1	    2	   3
#High Risk	      2908	2599	4670	2180
#Medium Risk	    5857	4023	4788	3583
#Low Risk	        16697	9943	31790	12728

######################################End of Assignment##################################
