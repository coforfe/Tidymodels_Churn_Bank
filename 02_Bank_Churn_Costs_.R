#-- https://towardsdatascience.com/bank-customer-churn-with-tidymodels-part-2-decision-threshold-analysis-c658845ef1f

library(tidymodels) #ML Metapackage
library(probably) #Threshold Analysis
library(forcats) #Working with factors
library(patchwork) #ggplot grids
tidymodels_prefer()
options(yardstick.event_first = FALSE)
class_metric <- metric_set(accuracy, f_meas, j_index, kap, precision, sensitivity, specificity, mcc)


best_result <- wf_sample_exp %>% 
  extract_workflow_set_result("UPSAMPLE_Boosted_Trees") %>% 
  select_best(metric = 'j_index')
xgb_fit <- wf_sample_exp %>% 
  extract_workflow("UPSAMPLE_Boosted_Trees") %>% 
  finalize_workflow(best_result) %>%
  fit(training(cust_split))

xgb_fit %>% 
  predict(new_data = testing(cust_split), type = 'prob') %>% 
  bind_cols(testing(cust_split)) %>% 
  ggplot(aes(x=.pred_1, fill = Exited, color = Exited)) +
  geom_histogram(bins = 40, alpha = 0.5) +
  theme_minimal() +
  scale_fill_viridis_d(aesthetics = c('color', 'fill'), end = 0.8) +
  labs(title = 'Distribution of Prediction Probabilities by Exited Status', x = 'Probability Prediction', y = 'Count')

#Generate Probability Prediction Dataset
xgb_pred <- xgb_fit %>% 
  predict(new_data = testing(cust_split), type = 'prob') %>% 
  bind_cols(testing(cust_split)) %>% 
  select(Exited, .pred_0, .pred_1)
#Generate Sequential Threshold Tibble
threshold_data <- xgb_pred %>% 
  threshold_perf(truth = Exited, Estimate = .pred_1, thresholds = seq(0.1, 1, by = 0.01))
#Identify Threshold for Maximum J-Index
max_j_index <- threshold_data %>% 
  filter(.metric == 'j_index') %>% 
  filter(.estimate == max(.estimate)) %>% 
  select(.threshold) %>% 
  as_vector()
#Visualise Threshold Analysis
threshold_data %>% 
  filter(.metric != 'distance') %>% 
  ggplot(aes(x=.threshold, y=.estimate, color = .metric)) +
  geom_line(size = 2) +
  geom_vline(xintercept = max_j_index, lty = 5, alpha = .6) +
  theme_minimal() +
  scale_colour_viridis_d(end = 0.8) +
  labs(x='Threshold', 
       y='Estimate', 
       title = 'Balancing Performance by Varying Threshold',
       subtitle = 'Verticle Line = Max J-Index',
       color = 'Metric')


#Threshold Analysis by Several Classification Metrics
list(pred_df = list(pred_df = xgb_pred), 
     threshold = list(threshold = seq(0.03, 0.99, by = 0.01))) %>% 
  cross_df() %>% 
  mutate(pred_data = map2(pred_df, threshold, ~mutate(.x, .prob_class = as_factor(if_else(.pred_1 < .y , 0, 1)))),
         pred_data = map2(pred_data,  threshold, ~mutate(.x, .prob_metric = if_else(.pred_1 < .y , 0, 1))),
         pred_metric = map(pred_data, ~class_metric(.x, truth = Exited, estimate = .prob_class)),
         roc_auc = map(pred_data, ~roc_auc(.x, truth = Exited, estimate = .prob_metric)),
         pr_auc = map(pred_data, ~pr_auc(.x, truth = Exited, estimate = .prob_metric)),
         pred_metric = pmap(list(pred_metric, roc_auc, pr_auc),~bind_rows(..1,..2,..3))) %>%
  select(pred_metric, threshold) %>%                                                            
  unnest(pred_metric) %>%                                                                        
  ggplot(aes(x=threshold, y=.estimate, color = .metric)) +
  geom_line(size = 1) +
  scale_color_viridis_d() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45)) +
  facet_wrap(~.metric, nrow = 2) +
  labs(title = 'Impact of Decision Threshold on Classification Metrics', x= 'Threshold', y = 'Estimate', color = 'Metric')

#--- CLV
train %>% 
  mutate(CreditCardFees = HasCrCard*149,
         AccountFees = (NumOfProducts - HasCrCard)*99,
         CLV = CreditCardFees + AccountFees) %>% 
  ggplot(aes(CLV)) +
  geom_histogram() +
  theme_minimal() +
  labs(title = 'Distribution of Annual CLV', x='CLV', y = 'Count')

#--- Cost function
threshold_data %>% 
  filter(.metric %in% c('sens', 'spec')) %>% 
  pivot_wider(id_cols = .threshold, values_from = .estimate, names_from = .metric) %>% 
  mutate(Cost_FN = ((1-sens) * 510 * 149), 
         Cost_FP = ((1-spec) * 1991 * 99),
         Total_Cost = Cost_FN + Cost_FP) %>% 
  select(.threshold, Cost_FN, Cost_FP, Total_Cost) %>% 
  pivot_longer(2:4, names_to = 'Cost_Function', values_to = 'Cost') %>% 
  ggplot(aes(x = .threshold, y = Cost, color = Cost_Function)) +
  geom_line(size = 1.5) +
  theme_minimal() +
  scale_colour_viridis_d(end = 0.8) +
  labs(title = 'Threshold Cost Function', x = 'Threshold')

#--- Scenario Analysis - Minimising Cost or Maximising Differentiation
threshold_data %>% 
  filter(.metric %in% c('sens', 'spec')) %>% 
  pivot_wider(id_cols = .threshold, values_from = .estimate, names_from = .metric) %>% 
  mutate(Cost = ((1-sens) * 510 * 149) + ((1-spec) * 1991 * 99),
         j_index = (sens+spec)-1) %>% 
  ggplot(aes(y=Cost, x = .threshold)) +
  geom_line() +
  geom_point(aes(size = j_index, color = j_index)) +
  geom_vline(xintercept = 0.47, lty = 2) +
  annotate(x = 0.36, y=100000, geom = 'text', label = 'Best Class Differentiation\nJ-Index = 0.56,\nCost = $57,629,\nThreshold = 0.47') +
  geom_vline(xintercept = 0.69, lty = 2) +
  annotate(x = 0.81, y = 100000, geom = 'text', label = 'Lowest Cost Model\nJ-Index = 0.48,\nCost = $48,329,\nThreshold = 0.69') +    
  theme_minimal() +
  scale_colour_viridis_c() +
  labs(title = 'Decision Threshold Attrition Cost Function', 
       subtitle = 'Where Cost(FN) = $149 & Cost(FP) = $99',
       x = 'Classification Threshold', size = 'J-Index', color = 'J-Index')

#--- Confusion Matrix
t1 <- xgb_pred %>% 
  mutate(.pred = make_two_class_pred(.pred_0, levels(Exited), threshold = 0.5)) %>%
  conf_mat(estimate = .pred, Exited) %>% 
  autoplot(type = 'heatmap') + 
  scale_fill_gradient2() +
  labs(title = 'Default Decision Threshold = 0.50')
t2 <- xgb_pred %>% 
  mutate(.pred = make_two_class_pred(.pred_0, levels(Exited), threshold = 0.47)) %>%
  conf_mat(estimate = .pred, Exited) %>% 
  autoplot(type = 'heatmap') + 
  scale_fill_gradient2() +
  labs(title = 'With Adjusted Decision Threshold = 0.47')
t3 <- xgb_pred %>% 
  mutate(.pred = make_two_class_pred(.pred_0, levels(Exited), threshold = 0.69)) %>%
  conf_mat(estimate = .pred, Exited) %>% 
  autoplot(type = 'heatmap') + 
  scale_fill_gradient2() +
  labs(title ='With Adjusted Decision Threshold = 0.69')
t2 / t1 / t3 +
  plot_annotation(title = 'Confusion Matrices for UPSAMPLE_Boosted_Trees')



