## Metabolites in Health and Disease by Melba Torres

#### Setup - ignore ####

pacman::p_load(tidyverse, performance, ggpubr, ggplot2, janitor, visdat, skimr, caret, reshape2, glmnet, nestedcv, reshape2, ggfortify, mdatools, dplyr, ggvegan, vegan, corrr, ggcorrplot, mdatools, nestedcv)

fav_colors <- c("#e60200", "#e96000","#e94196","#ed5c9b","cornflowerblue", "#00cdff", "forestgreen","#3aaf85", "#9DC183")

theme_set(theme_linedraw())


## Data Wrangling and Exploration

# # The raw data file contains metabolite abundance measurements for 103 Holstein cows at the three previously mentioned time points. Each cow has a unique ID for identification. A total of 265 metabolites are included. Additionally, the dataset includes information on cow parity (primiparous or multiparous).
# 
# The data is organized so that each column represent a cow, with rows being either "factors" or metabolites.

##### Data Set-up - ignore ####
Raw_data <- read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vSGQ3ezyJwWItEr8E8Pj8jN0gK_LutC3SWRKItP-lLNKs2liSgzrQjvCVPr0GMTWrF-F8uki4kgXAdS/pub?output=csv", col_names = F, show_col_types = F)
Raw_mod <- Raw_data
colnames(Raw_mod) <- Raw_data[1,]
Raw_mod <- Raw_mod[-1,]
df <- Raw_data
df_names <- paste("M",sep="_", 
                df$X1[3:length(df$X1)])|>
  str_replace_all("-", "_") |>
  str_replace_all(" ","_")

tdf <- t(df) |>
  data.frame()

colnames(tdf) <- c("CowNumber", "Factor", df_names)

tdf <- tibble(tdf[-1,])

tdf <- separate_wider_delim(tdf, Factor, delim = " | ", names = c("Diagnosis", "Time", "Parity"))

tdf <- tdf %>%
  mutate(Diagnosis = str_remove_all(Diagnosis, "Group:"),
         Time = str_remove_all(Time, "Time:"),
         Parity= str_remove_all(Parity, "Parity:"),
         # Leave only the cow number
         CowNumber = str_remove_all(CowNumber, "_.*"))

tdf <- tdf |>
  mutate(across(!CowNumber & !Diagnosis & !Parity & !Time, .fns = as.numeric),
         Diagnosis = as.factor(Diagnosis),
         Time = factor(Time, ordered = T, levels =  c("Prepartum", "Calving", "Diagnosis")),
         Parity = as.factor(Parity),
         CowNumber = as.factor(CowNumber))

Cow_All <- tdf[rowSums(is.na(tdf[,5:269])) != ncol(tdf[,5:269]), ]

Cow_calving <- Cow_All |>
  filter(Time == "Calving")

# Log Transform
Calving_log <- cbind(Cow_calving[1:4],
                 log2(Cow_calving[5:ncol(Cow_All)]))

# Scale
Calving_scaled <- cbind(Calving_log[1:4],
                    scale(Calving_log[5:ncol(Cow_All)]))

# Save
Coded_calv <- Calving_scaled |>
  mutate(Diagnosis = as.numeric(Diagnosis)-1,
         Parity = as.numeric(Parity)-1)


####
#### PLS-DA Supervised Machine Learning #####

# PLS-DA is a supervised machine learning method, meaning it incorporates response variables (class labels) during model fitting⁶. This contrasts with unsupervised methods like PCA, which don't use class labels. Like PCA, PLS-DA reduces data dimensionality by identifying latent variables (components) that capture the most relevant information⁶. However, PLS-DA focuses retaining the mot covariance between response and predictor variables in it's principal components. PLS-DA does not require independence between predictor variables or assume a distribution for the data.⁶

# We need to use the "coded" data here and turn it into a matrix
cod_tmp <- as.matrix(Coded_calv[,c(-1,-3)])
resp <- Coded_calv$Diagnosis

# Obrain a random index for splitting the data into training and validation
set.seed(575)
ind <- sample(x = 1:(nrow(cod_tmp)-2),
              size = 80)

# Save data
train_matrix <- cod_tmp[ind,-1]
train_response <- resp[ind] == 1 # Save the response variable at "true" if it is Met, or False is it is "Con"

# Validation data
val_matrix <- cod_tmp[-ind,-1]
val_response <- resp[-ind] == 1 # Save the response variable at "true" if it is Met, or False is it is "Con"


# Calibrate model
set.seed(575)
m.all <- mdatools::plsda(train_matrix, 
                train_response, 
                4, cv = 1, 
                classname = "Met", 
                center = F)

summary(m.all) 
plot(m.all)
par(mfrow = c(1, 2))
plotRMSE(m.all)
plotXYResiduals(m.all)


# # Below are the results of this first fit, for the calibration, the cumulative explained variance on the x-axis is 8.56 while it is 45.26 on Y. This model selected a total of 1 component. Now, it appears that our model is over-fitting as the calibrated model achieves an accuracy of ~81% but the cross validation resulted in an overall accuracy of 67.5%. While it is expected the data to do a bit better on the training set, but the difference seems too large here.
# 
# | X   | X cumxpvar | Y cumexpvar | TP  | FP  | TN  | FN  | Spec. | Sens. | Accuracy |
# |:----|:-----------|:------------|:----|:----|:----|:----|:------|:------|:---------|
# | Cal | 8.56       | 45.26       | 30  | 10  | 35  | 5   | 0.778 | 0.857 | 0.812    |
# | Cv  | NA         | NA          | 24  | 15  | 30  | 11  | 0.667 | 0.686 | 0.675    |


# I decided to check if outliers were affecting the results. In the tutorial, Kucheryavskiy references the paper titled ["Detection of Outliers in Projection-Based Modeling"](https://pubs.acs.org/doi/10.1021/acs.analchem.9b04611) by Rodionova and Pomerantsev⁹. Their approach consists of identifying outliers and removing them from the calibration set, and then re-fitting the model, this is repeated until there are no outliers; then, the removed points are predicted using the re-fitted model, and if the residuals are not outliers, then add them back into the calibration set and create a final model. Below is this process.


# Let's remove some outliers & repeat
  outliers <- which(categorize(m.all) == "extreme")

# keep data for outliers on a separate matrix
Xo <- train_matrix[outliers, , drop = FALSE]
yo <- train_response[outliers]

# remove data for outliers from training data
X <- train_matrix[-outliers,]
y <- train_response[-outliers] 

# make a new model for outlier free data #here
set.seed(575)
m.all <-  mdatools::plsda(X, y, 4, cv = 1, classname = "Met", center = F)

# Let's repeat the process until we have no outliers

###
# 1
###

# Let's remove some outliers & repeat
  outliers <- which(categorize(m.all) == "extreme")

# keep data for outliers on a separate matrix
Xo <- rbind(Xo, train_matrix[outliers, , drop = FALSE])
yo <- append(yo, train_response[outliers])

# remove data for outliers from training data
X <- X[-outliers,]
y <- y[-outliers] 

# make a new model for outlier free data
set.seed(575)
m.all <-  mdatools::plsda(X, y, 4, cv = 1, classname = "Met", center = F)

# Check for outliers again
which(categorize(m.all) == "extreme")

###
# 2
###

# Let's remove some outliers & repeat
outliers <- which(categorize(m.all) == "extreme")

# keep data for outliers on a separate matrix
Xo <- rbind(Xo, train_matrix[outliers, , drop = FALSE])
yo <- append(yo, train_response[outliers])

# remove data for outliers from training data
X <- X[-outliers,]
y <- y[-outliers] 

# make a new model for outlier free data
set.seed(575)
m.all <-  mdatools::plsda(X, y, 4, cv = 1, classname = "Met", center = F)

# Check for outliers again
which(categorize(m.all) == "extreme")

###
# 3
###

# Let's remove some outliers & repeat
outliers <- which(categorize(m.all) == "extreme")

# keep data for outliers on a separate matrix
Xo <- rbind(Xo, train_matrix[outliers, , drop = FALSE])
yo <- append(yo, train_response[outliers])

# remove data for outliers from training data
X <- X[-outliers,]
y <- y[-outliers] 

# make a new model for outlier free data
set.seed(575)
m.all <-  mdatools::plsda(X, y, 4, cv = 1, classname = "Met", center = F)

# Check for outliers again
which(categorize(m.all) == "extreme")

# We have removed all of the outliers 

summary(m.all)

#This model without outliers seems to be a better fit as it performs better on the "new" data. Next, let's predict the outliers with the model

### Now predict them
set.seed(575)
res <- predict(m.all, Xo, yo)


# Now plot the residuals for the predicted outliers 
plotXYResiduals(m.all, res = list("cal" = m.all$res$cal, 
                                  "out" = res),
                show.labels = TRUE)

out_ind <- c(-1,-3,-14)

plotXYResiduals(m.all, 
                res = list("cal" = m.all$res$cal, "out" = res),
                show.labels = TRUE, labels = "indices")


# Oh there's as issue here! Per the paper, the next step would be to add back the outliers whose predicted values do not end up being outliers themselves, unfortunately, after working on this for a while, I have not been able to actually figure out which ones those are, I have them labeled but it doesn't seem like the label actually corresponds to the index of the sample - I will look more into this since I do want to use this method for my own research, but for now there's just not enough time to figure it out. For the following prediction, I will use the model that does not include any outliers - which is acknowledge is not the recommended approach.
# 
# #### PLS-DA Prediction
# 
# Now we test the final model to predict the validation set and evaluate it's performance


set.seed(575)
m.pred <- predict(m.all, val_matrix, val_response)
summary(m.pred)

par(mfrow = c(2, 2))
plotSpecificity(m.pred)
plotSensitivity(m.pred)
plotMisclassified(m.pred)
plotPredictions(m.pred)
par(mfrow = c(1, 1))

plotPredictions(m.pred)


# This PLS-DA achieved ~61% accuracy when put to test on the validation set, this is only slightly better than what we would expect from simply guessing "Met" with a probability equal to its proportion in the data (\~50%). In other words, the model struggled to accurately predict for cows it hadn't encountered during training. This suggests limited generalizability and does raise concerns about the model's effectiveness for classifying unseen Holstein cow samples.
# 
# Despite the low predictive performance, some insights might still be gained from the analysis in the form of variable importance in projection (VIP) scores. As briefly discussed above, these scores highlight which metabolic features the model considers most influential in discriminating between "Met" and "Non-Met" states. However, given the model's low accuracy, it's important to interpret these VIP scores with caution as they may primarily reflect the specific characteristics of the training data (these Holstein cows) rather than providing reliable insights applicable to the broader Holstein population.
# 
# #### PLS-DA VIP Scores
# 
# Here, I extract the VIP scores for all input variables in the PSL-DA model (parity & all metabolites), then, I save the name and score of those exceeding a threshold of 1 (a common criterion for identifying important variables). This data frame will re-apear later in the project (see log fold change)


vipscore <- as.data.frame(vipscores(m.all)) |>
  rownames_to_column("Variable") |>
  arrange(desc(Met))

vipscore

nrow(vipscore[vipscore$Met > 1,])
# There are 85 metabolites with a VIP score greater than 1, these will be saved as a dataframe called VIPdf

VIPdf <- vipscore[vipscore$Met > 1,]

### For GLM with Elastic Net please see pdf or qmd file. 
### For references please see pdf or qmd file. 

####