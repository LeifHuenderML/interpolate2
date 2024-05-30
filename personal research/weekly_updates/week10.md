# Week 10 Research

## Weekly Strategy
- try to get a first draft of the paper written by the end of the week
- get at least a few predictive models up and running
- analyze the difference in predictive quality between the models with dimensionality reduction and the ones without

- for dimensionality reduction i will be using the 7 similarity scoring metrics that i have created to reduce the dimensionality of the data. since i have similarity scores measure similarity between 2 different geo locations, for each of their features, will use each individual features sim score as an output for the yield, as well as the average of all the sim scores features for the output of the yield. i will do this accross all 7 sim scores. Then i will do an average of all sim scores. then i will analyze this to see if their is better predictive ability from that than to a regular larger model trained on the higher dimensionality data. 


## Long-term Strategy

- Outcome: Comparative Climate Analysis:
  A. Analyze individual microclimate data streams to compare and calculate similarity scores.
  B. Assess combined microclimate data streams for seasonal variations comparison.
  C. Group years based on microclimate similarities to identify patterns.
  D. Develop predictive models (LSTM/transformer-based) linking climate to observed outcomes.
  E. Test models against various seasonal stages to evaluate performance.
  F. Synthesize findings to derive actionable insights.

## TODO

- [] look over the paper guidelines
- [] creaete an outline for the paper
- [] finish the research that will be needed for the paper
- [] start writing the paper


# Ideas

# Notes 

# Summary

## Tuesday:
- was at moscows campus all day geting filled in on the summer research fellowship

## Wednesday:
- completed the CITI training

## Thursday: 
- started in the morning by cleaning up the repo to make it easier to navigate
- added dates to the datasets for easier tracking
- added a readme to the datasets folder to make it easier to navigate
- created the ipynbs for touch svm.ipynb gpr.ipynb descision_trees.ipynb rndm_forest.ipynb xg_boost.ipynb kNN.ipynb naive_bayes.ipynb
- fixed a memory + space issue with the repo
- i realized that my similarity scores data is not very well labeled, so i will need to go back and relabel the data
- built the [sim_score_and_yield_05_23_24](../../data/pecan/sim_score_and_yield_05_23_24.csv) dataset
- started testing the quality of predictions from the dataset above using a svm but the results were not very good
- i then created a xgboost model and the results were slightly better but not very impressive
- inside the svm.ipynb is where i created the dataset of sim scores and yield
- i created a dummy model that just predicts the average of the yield for the training data, and the results were not very good at a mse of 484
- then testing that up against the xgboost model, the mse was 478 so there was slight improvement but not very impressive
- i think now what i want to do is to train a larger sequential model so to see if there is stronger predicitve ability from the larger model
- then i can think of a way to reduce the dimensionality of the data to see if that will improve the predictive ability of the model
- built and trained a lstm on weather sequences to output the yield, the results are not yet confirmed but look better than the sim scores by a 97% improvemnent

## Friday:
- started the day by writing a parralel processing function to train the lstms on to find the best hyperparameters faster
- added an early stopping callback to the lstm to stop training when the validation loss stops decreasing
- ran into difficulty tying to get the parallel processing to work, so i will need to come back to that later if need be
- i created a nice semi modular framework for training and testing lstms over a ranges of hyperparameters
- i began training the lstm script to find the best model from 864 different models, which will take a while to train
- the results are allraedy looking promising with the best model so far having a mape of 6%


## Saturday:
- Training the lstms models completed 
- i found the best lstm to have a input size of 128, i trained it on 32 feature sequences in batches of 32, 
- the model perfomed better than the sim scores model that i have tested so far with the sim scores model having a rmse of 484 and the lstm having a rmse of 145 suggesting a 70% improvement in predictive ability
- i wend through an created some visualiztions of the lstm
- i began working on outlining the paper, i know there will need to be more work done on the sim scores for it
- i still need to think  more about how i can use sim scores in a valuable way, or if there even is a way to use them.
- Started writing the paper
- got finished with the introduction 
- started on writing the methodologies 
- wrote the data sources and prepearation
- created a new notebook for makeing the first graph of the paper called [new mexicon pecan yield vis](../../src/data_visualization/new_mexico_pecan_yield_visualization.ipynb)
- finished writing the majority of the data secson for the paper

## Sunday:
- idea: is to craete sim scores for each day baset on location taht compares to each others location but instead of it being 6 dims it is 3 ie location of the yield output with a sim score against other location 1 -3 
- idea: second sim score model is to increase dimensionality by addind the 3 dims to that days weather data and train another model with that
- todo add readme for valley data 
- i started off by gathering the data for the valley feaver so  that we can start aquiring that weather data, i think that it would be best to get as much data as possible especially since i have a research budget for it
- i then continued writning the paper for my research i finished writing the first draft for the data preperation, i also wrote the first draft for the similarity scoring
- i created a new datavisualization notebook that created a heatmap of the similarity scores matching the color theme of the paper

## Monday: 
- created some more visualizations for the paper
- wrote the rest of the data prep for the paper
- worte the similarity analysis for the paper
- wrote the baseilen regressor for the paper 
- wrote the mlp regressor for the paper
- wote the xgboost fo the paper
- wrote the lstm for the paper
- worte the results 
- wrote the contluction 
- trained anothe nn on dataset 4 for the paper
- trained another lstm on dataset 4 for the paper
- trained xgboost on dataset 4 for the paper
- organized the codebase some 

# Next Week's Outline For 
- include all references into the paper
- includ the adknowledgements 
- format the paper
- edit it like crazy 


- propose for the next weeks of research to pivon to doing a study of th evallyey feaver the research question should be "can meteorological temporal data combinede with state of the art machine learning methods be used to predict early signs of valley fever cases' propose to use a lstm of transformer  to train on the weather data to then predict valley fever cases.