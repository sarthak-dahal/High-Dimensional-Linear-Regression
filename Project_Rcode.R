#Using High-Dimensional Linear Regression on Crime Rates in Communities Data
#
#Research Questions:
# RQ1: To what extent do ridge, lasso, and elastic net regression identify different
#      socioeconomic predictors of violent crime rates, and which predictors are
#      consistently selected across the three methods?
# RQ2: Which regularization method provides the best balance between predictive accuracy
#      and model complexity, based on test MSE and the number of retained predictors?
#
#Outline:
# 1. Data cleaning        2. Train/test split + standardization    3. OLS baseline
# 4. Ridge                5. Lasso                                  6. Elastic net (tuned alpha)
# 7. Comparison table     8. RQ1: variable selection / ranking      9. RQ2: repeated splits
# 10. Selection stability 11. Sensitivity check (log response)      12. Answers to RQs
#All figures are saved to the figures/ folder.

#HOW TO RUN: put this script and crimedata.csv in the same folder, then set R's working
#directory to that folder (RStudio: Session > Set Working Directory > To Source File Location)
#and click Source.Figures/tables are saved to figures/.

#Libraries Used
library(tidyverse)
library(glmnet)

dir.create("figures", showWarnings = FALSE)

#Helper: draw a base-R plot on screen AND save a PNG copy for the report
show_and_save <- function(file, plot_fun) {
  plot_fun()
  png(file.path("figures", file), width = 1600, height = 1100, res = 200)
  plot_fun()
  dev.off()
}

#Helper: coefficients (without intercept) of a cv.glmnet fit at "lambda.min" or "lambda.1se"
coef_vec <- function(cv, rule) {
  cf <- coef(cv, s = rule)
  setNames(as.numeric(cf[-1, 1]), rownames(cf)[-1])
}

#Helper: test-set accuracy measures
test_metrics <- function(y, pred) {
  mse <- mean((y - pred)^2)
  tibble(Test_MSE = mse, Test_RMSE = sqrt(mse), Test_R2 = 1 - mse / mean((y - mean(y))^2))
}


########################################################################################
# 1. DATA CLEANING
########################################################################################

#Importing the Data
crime <- read.csv("crimedata.csv",
                  na.strings = c("?", "NA"),
                  fileEncoding = "latin1")
dim(crime)

#Removing Descriptor Columns (community name, state, county code, community code, fold).
#Dropped by position because the first column name contains a stray encoding character.
names(crime)[1:5]
crime <- crime %>%
  select(-(1:5))

#Removing Redundant Columns: individual crime counts/rates that make up the response
#(would leak the answer) and the non-violent crime rate (a different outcome)
crime <- crime %>%
  select(-c("murders", "murdPerPop", "rapes", "rapesPerPop",
            "robberies", "robbbPerPop", "assaults", "assaultPerPop",
            "burglaries", "burglPerPop", "larcenies", "larcPerPop",
            "autoTheft", "autoTheftPerPop", "arsons", "arsonsPerPop", "nonViolPerPop"))

#Handling Missingness
sum(is.na(crime))

missing_summary <- crime %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "n_missing") %>%
  filter(n_missing > 0) %>%
  mutate(pct_missing = round(100 * n_missing / nrow(crime), 1)) %>%
  arrange(desc(n_missing))
print(missing_summary, n = Inf)

#The LEMAS police variables are missing for most communities (~85%), so imputing them
#would be mostly guesswork. I drop these columns and then drop the few remaining rows
#with missing values (mostly a missing response).
crime <- crime %>%
  select(-c("LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop",
            "LemasTotalReq", "LemasTotReqPerPop","PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol",
            "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor",
            "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "PolicCars",
            "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "PolicBudgPerPop", "LemasPctOfficDrugUn"))

n_before <- nrow(crime)
crime <- crime %>%
  drop_na()
cat("Rows before drop_na():", n_before, "| after:", nrow(crime),
    "| removed:", n_before - nrow(crime), "\n")
dim(crime)
sum(is.na(crime))

#Response variable: violent crimes per 100K population
summary(crime$ViolentCrimesPerPop)
p_resp <- ggplot(crime, aes(x = ViolentCrimesPerPop)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Violent Crimes per 100K Population",
       subtitle = "The response is strongly right-skewed",
       x = "Violent crimes per 100K", y = "Number of communities") +
  theme_minimal()
print(p_resp)
ggsave("figures/01_response_histogram.png", p_resp, width = 7, height = 4.5, dpi = 200)


########################################################################################
# 2. SPLITTING THE DATA INTO TRAIN AND TEST + STANDARDIZING
########################################################################################

#Written as a function so the exact same steps can be repeated over many random splits (Section 9).
#Predictors are standardized with the TRAINING means/sds only (no information from the test set).
#Because every predictor is on the same scale (1 SD), coefficient magnitudes are directly
#comparable, which is what lets me rank predictors by |coefficient| (Section 8).
prepare_split <- function(data, seed, train_frac = 0.8, nfolds = 10) {
  set.seed(seed)
  train_idx <- sample(nrow(data), size = floor(train_frac * nrow(data)))
  crime_train <- data %>% slice(train_idx)
  crime_test  <- data %>% slice(-train_idx)

  x_train_raw <- crime_train %>% select(-ViolentCrimesPerPop)
  x_test_raw  <- crime_test %>% select(-ViolentCrimesPerPop)

  train_means <- apply(x_train_raw, 2, mean)
  train_sds   <- apply(x_train_raw, 2, sd)

  list(x_train = as.matrix(scale(x_train_raw, center = train_means, scale = train_sds)),
       x_test  = as.matrix(scale(x_test_raw,  center = train_means, scale = train_sds)),
       y_train = crime_train$ViolentCrimesPerPop,
       y_test  = crime_test$ViolentCrimesPerPop,
       #same CV folds are used for every method so CV errors are directly comparable
       foldid  = sample(rep(1:nfolds, length.out = nrow(crime_train))))
}

split_main <- prepare_split(crime, seed = 4250)
x_train <- split_main$x_train
x_test  <- split_main$x_test
y_train <- split_main$y_train
y_test  <- split_main$y_test
foldid  <- split_main$foldid
n_pred  <- ncol(x_train)
cat("Training obs:", nrow(x_train), "| Test obs:", nrow(x_test), "| Predictors:", n_pred, "\n")


########################################################################################
# 3. OLS BASELINE (no regularization), fit on the training data only
########################################################################################

ols_fit  <- lm(y ~ ., data = data.frame(y = y_train, x_train))
summary(ols_fit)$r.squared
ols_pred <- predict(ols_fit, newdata = data.frame(x_test))
ols_metrics <- test_metrics(y_test, ols_pred)
ols_metrics

#Multicollinearity check: many predictors are near-duplicates of each other,
#which is the main reason regularization is needed here
cor_mat <- cor(x_train)
high_cor_pairs <- which(abs(cor_mat) > 0.9 & upper.tri(cor_mat), arr.ind = TRUE)
cat("Number of predictor pairs with |correlation| > 0.9:", nrow(high_cor_pairs), "\n")


########################################################################################
# 4. RIDGE REGRESSION (alpha = 0)
########################################################################################

ridge_fit <- glmnet(x_train, y_train, alpha = 0, standardize = FALSE)
ridge_cv  <- cv.glmnet(x_train, y_train, alpha = 0, foldid = foldid,
                       type.measure = "mse", standardize = FALSE)

show_and_save("02_ridge_path.png", function() {
  plot(ridge_fit, xvar = "lambda", label = FALSE, main = "Ridge Coefficient Paths\n")
  abline(v = log(ridge_cv$lambda.min), lty = 2, col = "red")
  abline(v = log(ridge_cv$lambda.1se), lty = 3, col = "blue")
  legend("topright", c("lambda.min", "lambda.1se"), lty = c(2, 3), col = c("red", "blue"), bty = "n")
})
#CV curve with +/- 1 SE error bars; left dashed line = lambda.min, right = lambda.1se
show_and_save("03_ridge_cv.png", function() {
  plot(ridge_cv, main = "Ridge: 10-fold CV MSE\n")
})

ridge_cv$lambda.min
ridge_cv$lambda.1se
min(ridge_cv$cvm)


########################################################################################
# 5. LASSO REGRESSION (alpha = 1)
########################################################################################

lasso_fit <- glmnet(x_train, y_train, alpha = 1, standardize = FALSE)
lasso_cv  <- cv.glmnet(x_train, y_train, alpha = 1, foldid = foldid,
                       type.measure = "mse", standardize = FALSE)

show_and_save("04_lasso_path.png", function() {
  plot(lasso_fit, xvar = "lambda", label = FALSE, main = "Lasso Coefficient Paths\n")
  abline(v = log(lasso_cv$lambda.min), lty = 2, col = "red")
  abline(v = log(lasso_cv$lambda.1se), lty = 3, col = "blue")
  legend("topright", c("lambda.min", "lambda.1se"), lty = c(2, 3), col = c("red", "blue"), bty = "n")
})
#Numbers along the top of the CV plot = number of nonzero coefficients at that lambda
show_and_save("05_lasso_cv.png", function() {
  plot(lasso_cv, main = "Lasso: 10-fold CV MSE\n")
})

lasso_cv$lambda.min
lasso_cv$lambda.1se
min(lasso_cv$cvm)


########################################################################################
# 6. ELASTIC NET REGRESSION, tuning alpha as well as lambda
########################################################################################

#alpha = 0 is ridge and alpha = 1 is lasso; elastic net is anything in between.
#Instead of fixing alpha = 0.5, I choose alpha by CV using the SAME folds.
alpha_grid <- seq(0, 1, by = 0.1)
alpha_cvs  <- map(alpha_grid, ~ cv.glmnet(x_train, y_train, alpha = .x, foldid = foldid,
                                          type.measure = "mse", standardize = FALSE))
alpha_tbl <- tibble(alpha      = alpha_grid,
                    lambda_min = map_dbl(alpha_cvs, "lambda.min"),
                    CV_MSE     = map_dbl(alpha_cvs, ~ min(.x$cvm)),
                    CV_SE      = map_dbl(alpha_cvs, ~ .x$cvsd[which.min(.x$cvm)]),
                    n_kept     = map_int(alpha_cvs, ~ sum(coef_vec(.x, "lambda.min") != 0)))
alpha_tbl

#Best elastic net alpha among the "true" mixtures (0 < alpha < 1)
en_rows    <- which(alpha_grid > 0 & alpha_grid < 1)
best_idx   <- en_rows[which.min(alpha_tbl$CV_MSE[en_rows])]
best_alpha <- alpha_grid[best_idx]
elastic_cv   <- alpha_cvs[[best_idx]]
elastic05_cv <- alpha_cvs[[which(abs(alpha_grid - 0.5) < 1e-8)]]
cat("Best elastic net alpha by CV:", best_alpha, "\n")

p_alpha <- ggplot(alpha_tbl, aes(x = alpha, y = CV_MSE)) +
  geom_errorbar(aes(ymin = CV_MSE - CV_SE, ymax = CV_MSE + CV_SE), width = 0.02, color = "grey60") +
  geom_line() + geom_point(size = 2) +
  geom_point(data = filter(alpha_tbl, alpha == best_alpha), color = "red", size = 4) +
  labs(title = "Choosing alpha: CV MSE across the ridge-to-lasso spectrum",
       subtitle = "alpha = 0 is ridge, alpha = 1 is lasso; red = best elastic net alpha; bars = +/- 1 SE",
       x = expression(alpha), y = "Minimum 10-fold CV MSE") +
  theme_minimal()
print(p_alpha)
ggsave("figures/06_alpha_tuning.png", p_alpha, width = 7, height = 4.5, dpi = 200)

elastic_fit <- glmnet(x_train, y_train, alpha = best_alpha, standardize = FALSE)
show_and_save("07_elastic_path.png", function() {
  plot(elastic_fit, xvar = "lambda", label = FALSE,
       main = paste0("Elastic Net Coefficient Paths (alpha = ", best_alpha, ")\n"))
  abline(v = log(elastic_cv$lambda.min), lty = 2, col = "red")
  abline(v = log(elastic_cv$lambda.1se), lty = 3, col = "blue")
  legend("topright", c("lambda.min", "lambda.1se"), lty = c(2, 3), col = c("red", "blue"), bty = "n")
})
show_and_save("08_elastic_cv.png", function() {
  plot(elastic_cv, main = paste0("Elastic Net (alpha = ", best_alpha, "): 10-fold CV MSE\n"))
})

elastic_cv$lambda.min
elastic_cv$lambda.1se
min(elastic_cv$cvm)


########################################################################################
# 7. COMPARISON TABLE (single train/test split)
########################################################################################

#For each method I report two lambda choices:
#  lambda.min = lambda with the lowest CV MSE (best accuracy)
#  lambda.1se = largest lambda within 1 SE of the minimum (simpler model, about as accurate)
summarise_cv <- function(method, alpha, cv, rule) {
  pred <- predict(cv, newx = x_test, s = rule)
  bind_cols(tibble(Method = method, Alpha = alpha, Lambda_rule = rule,
                   Lambda = cv[[rule]],
                   CV_MSE = cv$cvm[cv$lambda == cv[[rule]]]),
            test_metrics(y_test, pred),
            tibble(Predictors_kept = sum(coef_vec(cv, rule) != 0)))
}

model_list <- list(list("Ridge", 0, ridge_cv),
                   list("Lasso", 1, lasso_cv),
                   list("Elastic Net (alpha = 0.5)", 0.5, elastic05_cv),
                   list(paste0("Elastic Net (tuned alpha = ", best_alpha, ")"), best_alpha, elastic_cv))

results <- bind_rows(
  bind_cols(tibble(Method = "OLS (no penalty)", Alpha = NA, Lambda_rule = "-", Lambda = 0, CV_MSE = NA),
            ols_metrics,
            tibble(Predictors_kept = sum(!is.na(coef(ols_fit)[-1])))),
  map_dfr(model_list, function(m) {
    bind_rows(summarise_cv(m[[1]], m[[2]], m[[3]], "lambda.min"),
              summarise_cv(m[[1]], m[[2]], m[[3]], "lambda.1se"))
  })
)
print(results %>% mutate(across(where(is.numeric), ~ round(.x, 3))), width = Inf)
write_csv(results, "figures/results_single_split.csv")


########################################################################################
# 8. RQ1: WHICH PREDICTORS DOES EACH METHOD IDENTIFY?
########################################################################################

#Ridge never sets coefficients exactly to zero, so it does not select variables by itself.
#I RANK predictors by |coefficient| for every method
#(valid because all predictors are standardized). To compare ridge with the methods that do
#select variables, I use a cutoff of k = the number of predictors lasso keeps at lambda.min,
#so all three methods are compared on lists of the same length.
coef_tbl <- tibble(variable      = colnames(x_train),
                   Ridge         = coef_vec(ridge_cv, "lambda.min"),
                   Lasso         = coef_vec(lasso_cv, "lambda.min"),
                   `Elastic Net` = coef_vec(elastic_cv, "lambda.min")) %>%
  mutate(Ridge_rank   = rank(-abs(Ridge), ties.method = "min"),
         Lasso_rank   = rank(-abs(Lasso), ties.method = "min"),
         Elastic_rank = rank(-abs(`Elastic Net`), ties.method = "min"))

k <- sum(coef_tbl$Lasso != 0)
cat("Cutoff k (number of predictors lasso keeps):", k, "\n")

ridge_topk  <- coef_tbl %>% filter(Ridge_rank <= k) %>% pull(variable)
lasso_sel   <- coef_tbl %>% filter(Lasso != 0) %>% pull(variable)
elastic_sel <- coef_tbl %>% filter(`Elastic Net` != 0) %>% pull(variable)

#Ridge ranking (top 20)
coef_tbl %>% arrange(Ridge_rank) %>% select(variable, Ridge, Ridge_rank) %>% head(20)

#Overlap between the three lists
overlap_counts <- tibble(
  Set = c("Ridge top-k", "Lasso selected", "Elastic net selected",
          "Ridge & Lasso", "Ridge & Elastic net", "Lasso & Elastic net", "All three"),
  Count = c(length(ridge_topk), length(lasso_sel), length(elastic_sel),
            length(intersect(ridge_topk, lasso_sel)), length(intersect(ridge_topk, elastic_sel)),
            length(intersect(lasso_sel, elastic_sel)),
            length(Reduce(intersect, list(ridge_topk, lasso_sel, elastic_sel)))))
overlap_counts

#Predictors consistently identified by all three methods
consistent_vars <- Reduce(intersect, list(ridge_topk, lasso_sel, elastic_sel))
consistent_tbl <- coef_tbl %>%
  filter(variable %in% consistent_vars) %>%
  mutate(Same_sign = sign(Ridge) == sign(Lasso) & sign(Lasso) == sign(`Elastic Net`),
         Avg_rank  = (Ridge_rank + Lasso_rank + Elastic_rank) / 3) %>%
  arrange(Avg_rank)
print(consistent_tbl %>% mutate(across(where(is.numeric), ~ round(.x, 1))), n = Inf, width = Inf)
write_csv(consistent_tbl, "figures/consistent_predictors.csv")

#Predictors only one method picks (where the methods disagree)
only_ridge   <- setdiff(ridge_topk, union(lasso_sel, elastic_sel))
only_lasso   <- setdiff(lasso_sel, union(ridge_topk, elastic_sel))
only_elastic <- setdiff(elastic_sel, union(ridge_topk, lasso_sel))
only_ridge; only_lasso; only_elastic

#Agreement over the FULL ranking (not just one cutoff): Spearman correlation of |coefficients|
rank_cor <- cor(abs(as.matrix(coef_tbl[, c("Ridge", "Lasso", "Elastic Net")])), method = "spearman")
round(rank_cor, 3)

#Figure: top 15 predictors for each method, highlighting the consistent ones
top_plot_df <- coef_tbl %>%
  select(variable, Ridge, Lasso, `Elastic Net`) %>%
  pivot_longer(-variable, names_to = "Method", values_to = "coef") %>%
  group_by(Method) %>%
  slice_max(abs(coef), n = 15, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(Method = factor(Method, levels = c("Ridge", "Lasso", "Elastic Net")),
         Consistent = ifelse(variable %in% consistent_vars, "Selected by all three", "Not consistent"),
         var_facet = reorder(paste(variable, Method, sep = "___"), abs(coef)))

p_top <- ggplot(top_plot_df, aes(x = coef, y = var_facet, fill = Consistent)) +
  geom_col() +
  geom_vline(xintercept = 0, color = "grey40") +
  facet_wrap(~ Method, scales = "free") +
  scale_y_discrete(labels = function(x) sub("___.*$", "", x)) +
  scale_fill_manual(values = c("Selected by all three" = "firebrick", "Not consistent" = "grey65")) +
  labs(title = "Top 15 Predictors by |Standardized Coefficient|",
       subtitle = "Change in violent crimes per 100K for a 1 SD increase in the predictor",
       x = "Coefficient", y = NULL, fill = NULL) +
  theme_minimal() + theme(legend.position = "bottom")
print(p_top)
ggsave("figures/09_top_predictors.png", p_top, width = 12, height = 6, dpi = 200)

#Why the methods can disagree: correlated predictors. Lasso tends to keep one variable from
#a correlated group and drop the rest, while ridge spreads weight across the whole group.
#Example: the family-structure variables
fam_vars <- intersect(c("PctKids2Par", "PctFam2Par", "PctYoungKids2Par", "PctTeen2Par", "PctKidsBornNeverMar"),
                      colnames(x_train))
round(cor(x_train[, fam_vars]), 2)
coef_tbl %>% filter(variable %in% fam_vars) %>% select(variable, Ridge, Lasso, `Elastic Net`)


########################################################################################
# 9. RQ2: ACCURACY vs COMPLEXITY OVER 25 RANDOM SPLITS
########################################################################################

#One train/test split can favor a method by chance. I repeat the whole pipeline
#(split -> standardize -> 10-fold CV -> test) on 25 different random splits.
#Elastic net uses the alpha chosen above. This takes a minute or two.
run_one_split <- function(seed) {
  s <- prepare_split(crime, seed)
  ols  <- lm(y ~ ., data = data.frame(y = s$y_train, s$x_train))
  ols_mse <- mean((s$y_test - predict(ols, newdata = data.frame(s$x_test)))^2)

  methods <- list(Ridge = 0, Lasso = 1, `Elastic Net` = best_alpha)
  glmnet_rows <- map_dfr(names(methods), function(m) {
    cv <- cv.glmnet(s$x_train, s$y_train, alpha = methods[[m]], foldid = s$foldid,
                    type.measure = "mse", standardize = FALSE)
    map_dfr(c("lambda.min", "lambda.1se"), function(rule) {
      cf <- coef_vec(cv, rule)
      tibble(seed = seed, Method = m, Lambda_rule = rule,
             Test_MSE = mean((s$y_test - predict(cv, newx = s$x_test, s = rule))^2),
             Predictors_kept = sum(cf != 0),
             coefs = list(cf))
    })
  })
  bind_rows(tibble(seed = seed, Method = "OLS", Lambda_rule = "-", Test_MSE = ols_mse,
                   Predictors_kept = n_pred, coefs = list(NULL)),
            glmnet_rows)
}

n_splits <- 25
rep_results <- map_dfr(4250 + 1:n_splits, run_one_split)

rep_summary <- rep_results %>%
  group_by(Method, Lambda_rule) %>%
  summarise(Mean_Test_MSE  = mean(Test_MSE),
            SD_Test_MSE    = sd(Test_MSE),
            Mean_Test_RMSE = mean(sqrt(Test_MSE)),
            Mean_Kept      = mean(Predictors_kept),
            SD_Kept        = sd(Predictors_kept),
            .groups = "drop") %>%
  arrange(Mean_Test_MSE)
print(rep_summary %>% mutate(across(where(is.numeric), ~ round(.x, 1))), width = Inf)
write_csv(rep_summary, "figures/results_repeated_splits.csv")

#Paired comparison (same splits for every method): is the best method really better?
paired <- rep_results %>%
  filter(Lambda_rule %in% c("lambda.min", "-")) %>%
  select(seed, Method, Test_MSE) %>%
  pivot_wider(names_from = Method, values_from = Test_MSE)
lasso_vs_en    <- t.test(paired$Lasso, paired$`Elastic Net`, paired = TRUE)
lasso_vs_ridge <- t.test(paired$Lasso, paired$Ridge, paired = TRUE)
en_vs_ridge    <- t.test(paired$`Elastic Net`, paired$Ridge, paired = TRUE)
lasso_vs_ols   <- t.test(paired$Lasso, paired$OLS, paired = TRUE)
pairwise_tests <- tibble(
  Comparison   = c("Lasso - Elastic Net", "Lasso - Ridge", "Elastic Net - Ridge", "Lasso - OLS"),
  Mean_Diff_MSE = c(lasso_vs_en$estimate, lasso_vs_ridge$estimate, en_vs_ridge$estimate, lasso_vs_ols$estimate),
  p_value       = c(lasso_vs_en$p.value, lasso_vs_ridge$p.value, en_vs_ridge$p.value, lasso_vs_ols$p.value),
  First_wins    = c(mean(paired$Lasso < paired$`Elastic Net`), mean(paired$Lasso < paired$Ridge),
                    mean(paired$`Elastic Net` < paired$Ridge), mean(paired$Lasso < paired$OLS)))
print(pairwise_tests %>% mutate(across(where(is.numeric), ~ signif(.x, 3))))

#Figure: test MSE distribution across splits
p_box <- rep_results %>%
  mutate(Model = ifelse(Lambda_rule == "-", Method, paste0(Method, "\n(", Lambda_rule, ")")),
         Model = reorder(Model, Test_MSE, median)) %>%
  ggplot(aes(x = Model, y = Test_MSE, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(width = 0.15, size = 0.8, alpha = 0.5) +
  labs(title = paste("Test MSE over", n_splits, "random 80/20 splits"),
       x = NULL, y = "Test MSE") +
  theme_minimal() + theme(legend.position = "none")
print(p_box)
ggsave("figures/10_test_mse_boxplot.png", p_box, width = 10, height = 5, dpi = 200)

#Figure: the accuracy vs complexity trade-off (bottom-left is best)
p_tradeoff <- rep_summary %>%
  mutate(Model = ifelse(Lambda_rule == "-", Method, paste0(Method, " (", Lambda_rule, ")"))) %>%
  ggplot(aes(x = Mean_Kept, y = Mean_Test_MSE, color = Method)) +
  #lasso & elastic net (and ridge & OLS) nearly overlap, so points are shifted slightly sideways
  aes(x = Mean_Kept + case_when(Method %in% c("Lasso", "Ridge") ~ -1.5, TRUE ~ 1.5)) +
  geom_errorbar(aes(ymin = Mean_Test_MSE - SD_Test_MSE / sqrt(n_splits),
                    ymax = Mean_Test_MSE + SD_Test_MSE / sqrt(n_splits)), width = 1.5) +
  geom_point(aes(shape = Lambda_rule), size = 4) +
  scale_x_continuous(expand = expansion(mult = 0.15)) +
  scale_y_continuous(expand = expansion(mult = 0.15)) +
  labs(title = "Accuracy vs Model Complexity",
       subtitle = paste("Averages over", n_splits, "splits; bars = +/- 1 SE; lower-left is better"),
       x = "Average number of predictors kept", y = "Average test MSE",
       shape = "Lambda rule") +
  theme_minimal()
print(p_tradeoff)
ggsave("figures/11_accuracy_vs_complexity.png", p_tradeoff, width = 9, height = 6, dpi = 200)


########################################################################################
# 10. SELECTION STABILITY ACROSS SPLITS (strengthens RQ1)
########################################################################################

#A predictor that is selected in nearly every split is a robust finding;
#one that is selected only sometimes may be an artifact of a particular split.
#Ridge: counted as "selected" if it is in the top-k by |coef|, k = lasso's count in that split.
stab_input <- rep_results %>% filter(Lambda_rule == "lambda.min", Method != "OLS")
lasso_k_by_seed <- stab_input %>% filter(Method == "Lasso") %>% select(seed, k = Predictors_kept)

stability <- stab_input %>%
  left_join(lasso_k_by_seed, by = "seed") %>%
  mutate(selected = pmap(list(Method, coefs, k), function(m, cf, kk) {
    if (m == "Ridge") names(cf)[rank(-abs(cf), ties.method = "first") <= kk] else names(cf)[cf != 0]
  })) %>%
  select(seed, Method, selected) %>%
  unnest(selected) %>%
  count(Method, variable = selected) %>%
  mutate(freq = n / n_splits) %>%
  select(-n) %>%
  pivot_wider(names_from = Method, values_from = freq, values_fill = 0) %>%
  mutate(Min_freq = pmin(Ridge, Lasso, `Elastic Net`)) %>%
  arrange(desc(Min_freq), desc(Lasso))

#"Robust consistent predictors": chosen by all three methods in at least 80% of splits
robust_vars <- stability %>% filter(Min_freq >= 0.8)
print(robust_vars, n = Inf)
write_csv(stability, "figures/selection_stability.csv")

p_stab <- stability %>%
  slice_max(Ridge + Lasso + `Elastic Net`, n = 25, with_ties = FALSE) %>%
  pivot_longer(c(Ridge, Lasso, `Elastic Net`), names_to = "Method", values_to = "freq") %>%
  mutate(variable = reorder(variable, freq, mean),
         Method = factor(Method, levels = c("Ridge", "Lasso", "Elastic Net"))) %>%
  ggplot(aes(x = Method, y = variable, fill = freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = scales::percent(freq, accuracy = 1)), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue", labels = scales::percent, limits = c(0, 1)) +
  labs(title = paste("Selection frequency over", n_splits, "random splits"),
       subtitle = "Ridge: in top-k by |coef| (k = lasso's count); Lasso/EN: nonzero coefficient",
       x = NULL, y = NULL, fill = "Selected") +
  theme_minimal()
print(p_stab)
ggsave("figures/12_selection_stability.png", p_stab, width = 8, height = 8, dpi = 200)


########################################################################################
# 11. SENSITIVITY CHECK: LOG-TRANSFORMED RESPONSE
########################################################################################

#The response is right-skewed, so a few very high-crime communities could drive the fit.
#Refit lasso on log(1 + y) and check whether the same predictors come out on top.
lasso_log_cv <- cv.glmnet(x_train, log1p(y_train), alpha = 1, foldid = foldid,
                          type.measure = "mse", standardize = FALSE)
lasso_log_coef <- coef_vec(lasso_log_cv, "lambda.min")
lasso_log_sel  <- names(lasso_log_coef)[lasso_log_coef != 0]
log_pred <- expm1(predict(lasso_log_cv, newx = x_test, s = "lambda.min"))
cat("Lasso on log(1+y): predictors kept =", length(lasso_log_sel),
    "| test MSE on original scale =", round(mean((y_test - log_pred)^2), 1),
    "| overlap with original lasso =", length(intersect(lasso_log_sel, lasso_sel)), "\n")

top10_orig <- names(sort(abs(coef_vec(lasso_cv, "lambda.min")), decreasing = TRUE))[1:10]
top10_log  <- names(sort(abs(lasso_log_coef), decreasing = TRUE))[1:10]
cat("Top-10 lasso predictors shared between original and log response:",
    length(intersect(top10_orig, top10_log)), "of 10\n")
intersect(top10_orig, top10_log)


########################################################################################
# 12. ANSWERS TO THE RESEARCH QUESTIONS (computed from the results above)
########################################################################################

cat("\n==================== RQ1 ====================\n")
cat("Ridge keeps all", n_pred, "predictors, so it was ranked by |coefficient| with cutoff k =", k,
    "(lasso's count).\n")
cat("Lasso selected", length(lasso_sel), "| Elastic net (alpha =", best_alpha, ") selected",
    length(elastic_sel), "| Ridge top-k:", length(ridge_topk), "\n")
cat("Predictors identified by all three methods (single split):", length(consistent_vars), "\n  ",
    paste(consistent_tbl$variable, collapse = ", "), "\n")
cat("All consistent predictors have the same sign in all three models:", all(consistent_tbl$Same_sign), "\n")
cat("Spearman rank correlation of |coef|: Ridge-Lasso =", round(rank_cor["Ridge", "Lasso"], 2),
    "| Ridge-EN =", round(rank_cor["Ridge", "Elastic Net"], 2),
    "| Lasso-EN =", round(rank_cor["Lasso", "Elastic Net"], 2), "\n")
cat("Robust predictors (chosen by all three methods in >= 80% of", n_splits, "splits):",
    nrow(robust_vars), "\n  ", paste(robust_vars$variable, collapse = ", "), "\n")
cat("Picked only by ridge:", length(only_ridge), "| only by lasso:", length(only_lasso),
    "| only by elastic net:", length(only_elastic), "\n")

cat("\n==================== RQ2 ====================\n")
best_row <- rep_summary %>% filter(Method != "OLS") %>% slice_min(Mean_Test_MSE, n = 1)
#Simplest model whose mean test MSE is within 1 SE of the best one
best_se  <- best_row$SD_Test_MSE / sqrt(n_splits)
parsimonious <- rep_summary %>%
  filter(Method != "OLS", Mean_Test_MSE <= best_row$Mean_Test_MSE + best_se) %>%
  slice_min(Mean_Kept, n = 1, with_ties = FALSE)
ols_row <- rep_summary %>% filter(Method == "OLS")
cat("Lowest average test MSE:", best_row$Method, "(", best_row$Lambda_rule, ") =",
    round(best_row$Mean_Test_MSE), "with", round(best_row$Mean_Kept, 1), "predictors on average\n")
cat("OLS (all", n_pred, "predictors) average test MSE =", round(ols_row$Mean_Test_MSE), "\n")
cat("Simplest model within 1 SE of the best:", parsimonious$Method, "(", parsimonious$Lambda_rule, ") =",
    round(parsimonious$Mean_Test_MSE), "with only", round(parsimonious$Mean_Kept, 1), "predictors\n")
cat("Lasso vs Elastic Net (lambda.min), paired t-test p-value =", signif(lasso_vs_en$p.value, 3), "\n")
cat("Lasso vs Ridge (lambda.min), paired t-test p-value =", signif(lasso_vs_ridge$p.value, 3), "\n")

