#-- https://towardsdatascience.com/bank-customer-churn-with-tidymodels-part-1-model-development-cdec4eeb131c

library(tidymodels)
library(themis) #Recipe functions to deal with class imbalances
library(tidyposterior) #Bayesian Resampling Comparisons
library(baguette) #Bagging Model Specifications
library(corrr) #Correlation Plots
library(readr) #Read .csv Files
library(magrittr) #Pipe Operators
library(stringr) #String Manipulation
library(forcats) #Handling Factors
library(skimr) #Quick Statistical EDA
library(patchwork) #Create ggplot Patchworks
library(GGally) #Pair Plots
options(yardstick.event_first = FALSE) #Evaluate second factor level as factor of interest for yardstick metrics
library(tictoc)


#--  https://www.kaggle.com/shivan118/churn-modeling-dataset 

train <- read_csv("./data/Churn_Modelling.csv") %>% 
  select(-c(Surname, RowNumber, CustomerId)) 

train$Exited <- as.factor(train$Exited)

skim(train)

viz_by_dtype <- function(x,y) {
  title <- str_replace_all(y,"_"," ") %>% 
    str_to_title()
  if ("factor" %in% class(x)) {
    ggplot(train, aes(x, fill = x)) +
      geom_bar() +
      theme_minimal() +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 8)) +
      scale_fill_viridis_d() +
      labs(title = title, y = "", x = "")
  }
  else if ("numeric" %in% class(x)) {
    ggplot(train, aes(x)) +
      geom_histogram()  +
      theme_minimal() +
      theme(legend.position = "none") +
      scale_fill_viridis_d() +
      labs(title = title, y = "", x = "")
  } 
  else if ("integer" %in% class(x)) {
    ggplot(train, aes(x)) +
      geom_histogram() +
      theme_minimal() +
      theme(legend.position = "none") +
      scale_fill_viridis_d() +
      labs(title = title, y = "", x = "")
  }
  else if ("character" %in% class(x)) {
    ggplot(train, aes(x, fill = x)) +
      geom_bar() +
      theme_minimal() +
      scale_fill_viridis_d() +
      theme(legend.position = "none",
            axis.text.x = element_text(angle = 45, hjust = 1),
            axis.text = element_text(size = 8)) +
      labs(title = title, y  = "", x = "")
  }
}
variable_list <- colnames(train) %>% as.list()
variable_plot <- map2(train, variable_list, viz_by_dtype) %>%
  wrap_plots(               
    ncol = 3,
    heights = 150,
    widths = 150)
variable_plot
ggsave("./charts/eda.png", dpi = 600)


ggpairs(train %>% 
          select(-c(HasCrCard,IsActiveMember,NumOfProducts, Gender, Geography)) %>% 
          drop_na() %>% 
          mutate(Exited = if_else(Exited == 1, "Y","N")), ggplot2::aes(color = Exited, alpha = 0.3)) + 
  scale_fill_viridis_d(end = 0.8, aesthetics = c("color", "fill")) + 
  theme_minimal() +
  labs(title = "Numeric Bivariate Analysis of Customer Churn Data")
ggsave("./charts/ggpairs_all.png", dpi = 600)

#--- Categorical
train %>% 
  mutate(Exited = as.factor(if_else(Exited == 1, "Y", "N")),
         HasCrCard = if_else(HasCrCard == 1, "Y", "N"),
         IsActiveMember = if_else(IsActiveMember == 1, "Y", "N"),
         NumOfProducts = as.character(NumOfProducts)) %>% 
  select(Exited,where(is.character)) %>% 
  drop_na() %>% 
  mutate(Exited = if_else(Exited == "Y",1,0)) %>% 
  pivot_longer(2:6, names_to = "Variables", values_to = "Values") %>% 
  group_by(Variables, Values) %>% 
  summarise(mean = mean(Exited),
            conf_int = 1.96*sd(Exited)/sqrt(n())) %>% 
  ggplot(aes(x = Values, y = mean, color = Values)) +
  geom_point() +
  geom_errorbar(aes(ymin = mean - conf_int, ymax = mean + conf_int), width = 0.1) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.title.x = element_blank(),
        axis.title.y = element_blank()) +
  scale_color_viridis_d(aesthetics = c("color", "fill"), end = 0.8) +
  facet_wrap(~Variables, scales = 'free') +
  labs(title = 'Categorical Variable Analysis', subtitle = 'With 95% Confidence Intervals')
ggsave("./charts/ggpairs_categoricals.png", dpi = 600)

#--- Modeling 
set.seed(246)
cust_split <- initial_split(train, prop = 0.75, strata = Exited)

dt_cust <- 
  decision_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")
rf_cust <- 
  rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>% 
  set_engine("ranger", importance = "impurity") %>% 
  set_mode("classification")
xgboost_cust <- 
  boost_tree(mtry = tune(), trees = tune(), min_n = tune(), tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(), sample_size = tune())  %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")
bagged_cust <- 
  bag_tree(cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# #-- Recipes
# recipe_template <-
#   recipe(Exited ~., data = training(cust_split)) %>% 
#   step_integer(HasCrCard, IsActiveMember, zero_based = T) %>% 
#   step_integer(NumOfProducts) %>% 
#   step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
#               CreditScoreAgeRatio = CreditScore/Age,
#               TenureAgeRatio = Tenure/Age,
#               SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>% 
#   step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>% 
#   step_dummy(all_nominal_predictors()) %>% 
#   step_samplingmethod(Exited) #Change or Add Sampling Steps Here as Necessary

recipe_1 <-
    recipe(Exited ~., data = training(cust_split)) %>%
    step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
    step_integer(NumOfProducts) %>%
    step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
                CreditScoreAgeRatio = CreditScore/Age,
                TenureAgeRatio = Tenure/Age,
                SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
    step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
    step_dummy(all_nominal_predictors()) %>%
    step_smote(Exited) 

recipe_2 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rose(Exited) 

recipe_3 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_bsmote(Exited) 

recipe_4 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_upsample(Exited) 

recipe_5 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_adasyn(Exited) 

recipe_6 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_tomek(Exited) 

recipe_8 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors())

recipe_7 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nearmiss(Exited) 


recipe_9 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(Exited) %>%
  step_downsample(Exited)

recipe_10 <-
  recipe(Exited ~., data = training(cust_split)) %>%
  step_integer(HasCrCard, IsActiveMember, zero_based = T) %>%
  step_integer(NumOfProducts) %>%
  step_mutate(SalaryBalanceRatio = EstimatedSalary/Balance,
              CreditScoreAgeRatio = CreditScore/Age,
              TenureAgeRatio = Tenure/Age,
              SalaryBalanceRatio = if_else(is.infinite(SalaryBalanceRatio),0,SalaryBalanceRatio)) %>%
  step_scale(all_numeric_predictors(), -c(HasCrCard, Age, IsActiveMember, NumOfProducts)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rose(Exited) %>%
  step_downsample(Exited)

#-- Corrrelation plot - Just for recipe-8 (no imbalance transformation)
cust_train <- recipe_8 %>% prep() %>% bake(new_data = NULL)
cust_test  <- recipe_8 %>% prep() %>% bake(testing(cust_split))
cust_train %>% 
  bind_rows(cust_test) %>% 
  mutate(Exited = as.numeric(Exited)) %>% 
  correlate() %>%
  rplot(print_cor = T, .order = "alphabet") +
  scale_color_gradient2(low = 'orange', high = 'light blue') + 
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Correlation Plot for Trained Dataset")
ggsave("./charts/Correlations_plot.png", dpi = 600)

#--- Workflow
recipe_list <- 
  list(
       SMOTE = recipe_1, ROSE = recipe_2,
       BSMOTE = recipe_3, UPSAMPLE = recipe_4, 
       ADASYN = recipe_5, TOMEK = recipe_6, 
       NEARMISS = recipe_7, NOSAMPLING = recipe_8, 
       SMOTEDOWNSAMPLE = recipe_9, ROSEDOWNSAMPLE = recipe_10
      )

model_list <- 
  list(Decision_Tree = dt_cust, Boosted_Trees = xgboost_cust, 
       Random_Forest = rf_cust, Bagged_Trees = bagged_cust)

wf_set <- 
  workflow_set(preproc = recipe_list, models = model_list, cross = T)

set.seed(246)
train_resamples <- 
  vfold_cv(training(cust_split), v = 5, strata = Exited)

class_metric <- metric_set(
  yardstick::accuracy, 
  yardstick::f_meas, 
  yardstick::j_index, 
  yardstick::kap, 
  yardstick::precision, 
  yardstick::sensitivity, 
  yardstick::specificity, 
  yardstick::mcc
)

tic() 
doParallel::registerDoParallel(cores = 6)
wf_sample_exp <- 
  wf_set %>% 
  workflow_map(resamples = train_resamples, 
               verbose = TRUE, 
               metrics = class_metric, 
               seed = 246)
toc()
#--- It takes -> 1.5 hours (Mac)

#--- Metrics Evaluation
collect_metrics(wf_sample_exp) %>% 
  separate(wflow_id, into = c("Recipe", "Model_Type"), sep = "_", remove = F, extra = "merge") %>% 
  group_by(.metric) %>% 
  select(-.config) %>% 
  distinct() %>%
  group_by(.metric, wflow_id) %>% 
  filter(mean == max(mean)) %>% 
  group_by(.metric) %>% 
  mutate(Workflow_Rank =  row_number(-mean),
         .metric = str_to_upper(.metric)) %>% 
  arrange(Workflow_Rank) %>% 
  ggplot(aes(x = Workflow_Rank, y = mean, color = Model_Type)) +
  geom_point(aes(shape = Recipe)) +
  scale_shape_manual(values = 1:n_distinct(recipe_list)) +
  geom_errorbar(aes(ymin = mean - std_err, ymax = mean + std_err)) +
  theme_minimal() +
  scale_color_viridis_d() +
  labs(title = "Performance Comparison of Workflows", x = "Workflow Rank", 
       y = "Error Metric", color = "Model Types", shape = "Recipes") +
  facet_wrap(~.metric,scales = 'free_y',ncol = 4)
ggsave("./charts/Performance_Comparison_Workflows.png", dpi = 500 )

#-- Bayesian
#- Execution Error...
jindex_model_eval <- 
  perf_mod(wf_sample_exp, metric = "j_index", iter = 5000)
jindex_model_eval %>% 
  tidy() %>% 
  mutate(model = fct_inorder(model)) %>% 
  separate(model, into = c("Recipe", "Model_Type"), sep = "_", remove = F, extra = "merge") %>% 
  ggplot(aes(x = posterior, fill = Model_Type)) +
  geom_density(aes(alpha = 0.7)) +
  theme_minimal() +
  scale_fill_viridis_d(end = 0.8) +
  facet_wrap(~Recipe, nrow = 10) +
  labs(title = "Comparison of Posterior Distributions of Model Recipe Combinations", 
       x = expression(paste("Posterior for Mean J Index")), 
       y = "")
ggsave("./charts/Bayesian_probabilities.png", dpi = 500 )

#------ END OF FILE -------
