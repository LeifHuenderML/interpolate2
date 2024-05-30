# Week 2 Research

## Morning Strategy

- Thoroughly review as many papers as I can over a 5-hour span.
- Come up with follow-up questions based on my readings.
- Develop different ideas for the implementation of various models.

## Long-term Strategy

- Last week, Dr. Everett and I discussed the desired outcomes of this research.
- Outcome: Comparative Climate Analysis:
  A. Analyze individual microclimate data streams to compare and calculate similarity scores.
  B. Assess combined microclimate data streams for seasonal variations comparison.
  C. Group years based on microclimate similarities to identify patterns.
  D. Develop predictive models (LSTM/transformer-based) linking climate to observed outcomes.
  E. Test models against various seasonal stages to evaluate performance.
  F. Synthesize findings to derive actionable insights.

## TODO

- Read through the papers to create follow-up questions, jot down insights, and strategize future steps.
- Narrow down the locations for crop yield studies.
- Narrow down the locations for malaria outbreak analysis.
- Acquire OpenWeather data for each selected location.
- Prepare the data using Pandas.
- Analyze the data using Matplotlib.
- Calculate similarity scores by analyzing unique geographic weather locations.

# Paper Review

## Key Papers I Read

### ClimaX: A Foundation Model for Weather and Climate

- It is a generalizable deep learning model for weather and climate science.
- Is trained using heterogeneous datasets spanning different variables, spatial-temporal coverages, and physical groundings.
- Is an extension of the transformer architecture that uses novel encoding and aggregation blocks to effectively use available compute while maintaining general utility.
- Pretrained with a self-supervised learning objective on climate datasets from the CMIP6.
- Can be fine-tuned to address many climate and weather tasks, including ones unseen in the pretraining phase.
- Beat benchmarks at the time for weather forecasting and climate projections.
- Source code is available on [ClimaX Github](https://github.com/microsoft/ClimaX).
- ClimaX proposes that using neural networks may be a better alternative to general circulation models.
- A general circulation model is one that uses differential equations to relate the flow of energy and matter in the environment.
- The downfall of GCMs is that they struggle to accurately represent physical processes and initial conditions at fine resolutions.
- GCMs are slow.
- The key idea in their paper and others using a similar methodology is to train deep neural networks to predict the target atmospheric variables using historic global datasets.
- Downside is they are not grounded to real physics, so they have the potential to hallucinate and make up relations that don't exist.
- They propose to increase generalizability to be less brittle when deployed on new data is to create a foundation model and then zero-shot train it to the task at hand.
- The key way to train a foundation model is to train it on a massive unsupervised dataset.
- Climate data is a multimodal task when you are trying to create a generalizable climate model.

## MMST-ViT: Climate Change-Aware Crop Yield Prediction via Multimodal Spatial Temporal Vision Transformer

- Date: 2023-2024 (date wasn't specified, but there was a citation in 2023 in their paper)
- Claim: Timely predicting crop yields remains challenging as crop growth is sensitive to growing season weather variations and climate change.
- They predict at the county level across the US by looking at the effects of short-term meteorological variation during the growing season and the long-term climate change on crops.
- Their model is comprised of a multimodal transformer, a spatial transformer, and a temporal transformer.
- Their model uses both visual remote sensing data and short-term meteorological data for modeling the effect of growing season weather variations on crop growth.
- The spatial transformer learns the high-resolution spatial dependency among counties for accurate agricultural tracking.
- The temporal transformer captures the long-range temporal dependency for learning the impact of long-term climate change on crops.
- They also made a new multimodal contrastive learning technique to pre-train their model without extensive human supervision (this will prove to be very valuable in our case).
- Dataset and source code are available [here](https://github.com/fudong03/MMST-ViT).
- DL-based solutions for crop yield predictions are grouped into two main categories:
    1. Remote sensing data-based
    2. Meteorological data-based approaches
- Both independently fail to perform as well because they miss key insights into the data. To get a clear picture of what is happening, using both is preferred.
- Their solution has two goals:
    1. Capture the impacts of both short-term growing season weather variations and long-term climate change on crops.
    2. Leverage high-resolution remote sensing data for accurate agricultural tracking.
- Their model outperforms the state-of-the-art counterparts.
- They used a vision transformer, which is an adaptation of a regular transformer but for computer vision.
- They used the USDA dataset for crop yield data.
- The HRRR dataset for meteorological data.
- The Sentinel-2 imagery for satellite imaging.
- ViT-based models are prone to overfit.
- Conventional pre-training methods like SimCLR only marginally improve crop yield prediction performance because they only consider the visual data, rendering it practically useless for multimodal models.
- Model operates on county-level data.
- All hidden layers of their model were set to the same size.
- They used the Pyramid Vision Transformer for the base architecture.

# Other Papers I Read That Had Value Related To Our Research

## Machine Learning Approaches for Crop Yield Prediction with MODIS and Weather Data

- Date: 2019
- They used a suite of different models to do essentially the same thing we are doing: they are taking weather data from MODIS and using that as an input to output yield predictions for corn in Iowa.
- They trained CNNs, ANNs, SSAEs, and LSTMs.
- They found the CNN model to be the most accurate over all the models they trained.

## Applications of Remote Sensing in Precision Agriculture: A Review

- Date: 2020
- Not very relevant to our research.

## Crop Yield Prediction Using Machine Learning: A Systematic Literature Review

- Date: 2020
- They did a literature review of 567 relevant studies that used ML to predict crop yield, from that they selected 50 that met their requirements, thoroughly analyzed them, and came up with future areas of research in this area.
- The most used features are temperature, rainfall, soil type, and the most applied algorithm is ANN.
- CNNs were the most widely used algorithms since the value of transformers had yet to be realized.
- Most of it is outdated since they didn't use transformers, but the methodologies are probably still good.
- They reviewed papers from 2008 to 2019.

## Forecasting Future Crop Suitability with Microclimate Data

- Date: 2021
- They used microclimate modeling techniques to generate 100m spatial resolution climate datasets for the southwest of the UK for present day and predicted 30 years in the future.
- Was used to predict 56 crop varieties.
- They state that increasing the availability of microclimate data has been identified as crucial for improving assessments of climate suitability for crops.
- For data, they used the daily min and max temp at 1km grid res.
- 6 hourly sea level pressure, wind speed and direction, and specific humidity available at 2-degree grid res from NOAA-NCEP.
- Hourly SIS and DNI radiation and cloud fractional cover available at 0.05-degree grid res from EUMETSAT.
- Daily mean sea surface temperatures at a grid res of 0.25-degree from NOAA.
- They focused on Cornwall and the Isles of Scilly.

## New Learning Approach for Unsupervised Neural Networks Model with Application to Agriculture Field

- Date: 2020
- Two new algorithms mentioned here that weren't anywhere else:
    1. The Kohonen Self-Organizing Map
    2. The Gram-Schmidt algorithm
- It seems like this is a current paper on past ideas, mentioning models that have not been state-of-the-art since 2008.

## Predicting Multidimensional Environmental Factor Trends in Greenhouse Microclimates Using Hybrid Ensemble Approach

- Date: 2023
- (Details to come back to.)

## Riemannian Manifold Learning for Nonlinear Dimensionality Reduction

- Date: 2006
- The framework for Riemannian manifold learning is formulated as constructing local coordinate charts for a Riemannian manifold.
- At the time of the paper, the most widely used was the Riemannian normal coordinates chart.
- Their paper proposes a more efficient approach.
- Their method is derived by reconstructing the manifold in the form of a simplicial complex whose dimension is determined as the max dimension of its simplices.

## Riemannian Diffusion Models

- Date: 2022
- Weather data may best be represented on a Riemannian manifold with non-zero curvature.
- Diffusion models have surpassed GANs because they don't have the issue of adversarial optimization.
- A diffusion model consists of a fixed Markov chain that progressively transforms data to a prior defined by the inference path, and a generative model which is another Markov chain that is learned to invert the inference process.
- Much of the current success for diffusion-based generative models is purpose-built for Euclidean spaces, which does not easily translate to Riemannian manifolds.
- This paper sets out to generalize conventional diffusion models on Euclidean spaces to arbitrary Riemannian manifolds.
- Their approach uses the Stratonovich SDE formulation for which the conventional chain rule of calculus holds.
- That can be exploited to define diffusion on a Riemannian manifold.
- This is a very math-heavy paper; to fully grasp what is happening, a lot more review on the math behind this is needed.
- Saving for down the road when we get to the point where we want to use a Riemannian model.

## Utilizing a Novel High-Resolution Malaria Dataset for Climate-Informed Predictions with a Deep Learning Transformer Model

- Date: 2023
- Their paper sets out to use a transformer on the same malaria outbreak data we are looking at to better predict outbreaks.
- Their transformer performed consistently with increased accuracy as more climate variables were used, indicating further potential for this prediction framework to predict malaria incidence at a daily level using climate data.
- By building accurate malaria predictors, we can go into areas of low data and eradicate the disease before it can spread.
- Main sources of malaria predictions are provided by statistical and conventional ML models, with accuracy varying from 70 to 90 percent.
- Disadvantages of statistical models are that they have a short prediction window and low temporal resolution.
- When DL models were used, they were found to tend to outperform traditional ML models.
- Downside of regular DL models is that they are not robust enough to handle sudden outbreak events instigated by non-seasonal climate variability.
- That is where transformers come in; they have shown to perform very strongly on time series forecasting.
- Their paper found there to be a lack of studies using transformers.

# Ideas

- I propose that we find the state-of-the-art climate model and fine-tune it to our data to see if it will make a good generalization, especially off of the smaller datasets like pecans. ClimaX was released by Microsoft in 2023; there may be more current models.
- Since it has already been done with corn in the paper "Machine Learning Approaches for Crop Yield Prediction with MODIS and Weather Data", I think that we should pivot back to a smaller dataset like pecans for the main focus of the study. I believe that it may produce more interesting results and be a bigger challenge to create a suite of models capable of generalizing to less data.
- We should use corn from Iowa as a comparison of model accuracy against the "Machine Learning Approaches for Crop Yield Prediction with MODIS and Weather Data" paper and see if our model generalizes better.
- "Machine Learning Approaches for Crop Yield Prediction with MODIS and Weather Data" did not use transformers, so that is something we have as an opportunity.
- My idea for the supervised portion of this project is to train a simple decision tree as a baseline, then a CNN because of their high performance and use on similar issues like in the paper "Crop Yield Prediction Using Machine Learning: A Systematic Literature Review", then our own transformer, then a foundation model fine-tuned, then a Riemann manifold model.
- Look into the Mamba model too; that may be valuable in our research.
- Look into Tide, a long time series model that outperforms transformers.
- See if we can get access to remote sensing imaging for data.
- Would using something like a GAN to generate pretraining data be valuable to help extrapolate a larger, higher-res model to the real data?
- Look into diffusion models.
- Look into Markov chains.
- Look into Markov decision processes.
- Try to find the next state-of-the-art algorithm like MAMBA and see if it can outperform transformers on our specific task.

# Summary

Today I feel was very productive. I reviewed 15 papers covering research on crop yield predicions, modelling microclimates, modelling climate change, predicting malaria outbreaks, Riemannian manifolds, and literature reviews of the methodologies currently being used in crop yield prediction. I gained numerous valuable insights into methods that could benefit our project moving forward. It appears that a common trend is shifting towards the use of transformers with labeled outputs for predicting various spatial-temporal microclimate data. I'm aware there's been some buzz around new models that are quite recent but seem capable of outperforming transformers in NLP tasks. One such model is MAMBA, which I am currently evaluating to determine its potential value for our project. I am eager to see us advance in microclimate modeling with a new state-of-the-art model, especially since there have already been over 50 valuable papers utilizing models like CNNs, RNNs, LSTMs, etc., on this issue. Moving forward, I propose that next week I obtain the data we should use from the weather database and begin preparing it and analyzing similarity scores. Towards the end of the day, I started to read more about Riemannian manifolds and came across a very dense, yet insightful paper titled "Riemann Diffusion Models." I definitely believe that exploring Riemannian models further could yield highly valuable results. I was also considering the possibility of incorporating satellite imaging data into our dataset, as several papers mentioned the value of having both meteorological data and image data for achieving more accurate results. Finally, I am interested in creating a generative model that produces similar but fictional data to feed into our predictive model first to pretrain it so that it won't immediately overfit on our data, especially on classifications that have very little information like pecans. I spent the rest of the day searching for more papers to fill the gaps in knowledge from what I've read, and I plan on reviewing those next week.

# Next Week's Outline For Work

- Complete the review of the collected papers.
- Acquire weather data for analysis. I am curious about our budget for this. Each location in the historical database is $10.
- Explore the possibility of obtaining satellite imaging data.
- Prepare datasets: weather, malaria, crop yield, and potentially imaging data.
- Begin analysis of climate data by calculating similarity scores.
- Continue to document all of my findings and have them ready to present to you at the end of day meeting.


https://medium.com/@geronimo7/mamba-a-shallow-dive-into-a-new-architecture-for-llms-54c70ade5957
