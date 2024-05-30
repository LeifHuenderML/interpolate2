# Project Overview
Introduction
Input Data:
Microclimate Data:

    URL: https://openweathermap.org/api
    Description: We plan to utilize the Open Weather API to acquire historical climate data for selected regions. This API provides extensive information on temperature, humidity, storms, wind direction and speed, sunlight, etc., with the potential for hourly data access in most locations.

Application Area 1: Malaria Outbreaks

    URL: https://apps.who.int/malaria/maps/threats/#/download
    Description: This resource offers historical data on malaria vector occurrences, including location (latitude and longitude) and counts of invasive mosquito species, with options to filter by location.

Application Area 2: Agricultural Yield Data

    URL: https://quickstats.nass.usda.gov/#4BAF38D1-435D-335E-8EA2-BC0B8ADE1833
    Description: Access to county-level agricultural yield data for various crops is available through this site. It allows for climate comparison on a county and annual basis by selecting a representative city for the county.

Project Phases:

    Initial Data Collection and Analysis:
        Selection of a commodity, specific counties within a state, and multiple years of yield data for those areas.
        Example: Analyzing Pecan yield in New Mexico, covering several counties and years, alongside microclimate data from February to November.

    Subset Locations for Malaria Outbreaks Analysis:
        Identifying specific locations for in-depth malaria outbreak analysis in conjunction with corresponding microclimate data collection.

    Comparative Climate Analysis:
    This phase aims to deepen our understanding of climate's impact on our areas of interest through a multifaceted approach:
        A. Analyze individual microclimate data streams to compare and calculate similarity scores.
        B. Assess combined microclimate data streams for comparing seasonal variations.
        C. Group years based on microclimate similarities to identify patterns.
        D. Develop predictive models (LSTM/transformer-based) linking climate to observed outcomes.
        E. Test models against various seasonal stages to evaluate performance.
        F. Synthesize findings to derive actionable insights.

# Leif Huender's Notes

Initially inclined to focus on acorns, I discovered limitations due to insufficient data. Further research led me to a more robust dataset for corn in Iowa. A decision is pending on data volume requirements.

## Further Steps

I will review relevant literature to enhance our project strategy.

## Game Plan

- Visualize the datasets for both pecan and corn, and another data point that has a lot of data.
- Completed visualization.

## What I Have Done

- Created a Conda environment to work with IPYNB files to create data visualizations.
- Created visualizations of the datasets with the most data in different areas.
- Conducted an in-depth analysis of the different datasets to determine what information can be gleaned.
- Collected papers to review about the different approaches to take.

## Questions I Have

### Unanswered



### Answered

- Is the microclimate data + malaria data + the yield data for the same locations? Answer: Yes, it is. The yield and malaria are going to be the output vector, and the input is going to be the microclimate data.
- How much data are we trying to analyze? It seems to me that to build a strong predictive model, we will need much more data on yield and so on. Answer: More is better.
- If I am not mistaken, there is no data in America regarding malaria outbreaks. :correct we are not doing malaria in the US we will find an area with a historical record of malaria and collect data from that area.

## Notes on Papers I Read

### Calculate Similarity — The Most Relevant Metrics in a Nutshell Notes

- Two different similarity metric groups:

#### Similarity-Based Metrics:

- **Pearson's Correlation** - Measures the relationship between two quantitative continuous variables.
- **Spearman's Correlation** - A non-parametric statistic that ranks the data instead of taking the original values, capable of identifying perfect relationships that are non-linear.
- **Kendall's Tau** - Similar to Spearman's correlation, a non-parametric measure of a relationship, has smaller variability when using larger sample sizes, less efficient but should not matter for our task.
- **Cosine Similarity** - Mentioned by Dr. Everett, popular in text analysis, good at comparing data irrespective of their size differences, used for comparing real values.
- **Jaccard Similarity** - Used for comparing two binary vectors, computationally more expensive because it matches all points from one set to another, good for detecting duplicates.

#### Distance-Based Metrics:

- **Euclidean Distance** - Measures the straight-line distance between two vectors.
- **Manhattan Distance** - In many ML applications, Euclidean distance is the metric of choice; for high-dimensional data, Manhattan is preferable because it yields more robust results.

- Similarity-based metrics determine the most similar objects, with the highest values implying they live in closer neighborhoods.

### A Machine Learning-Based Comparative Approach to Predict the Crop Yield Using Supervised Learning With Regression Models

- the goal stated in their abstract it to develop a ml model that predicts farm production
- they used supervised learning with six regression models
- their random forest regressor outperformed all the other models
- i think that this paper is out of date with the current modes of ml 
- they say it is a multi dimensioanal reggresion issue
- they also used openweather
- they had an accurracy of:
    - logistic regression 87.8% 
    - naive-bayes was 91.6%
    - random forsest was 92.8%
- this paper was more of a comparative summary of what others have done to build and test ml models on yield predictions
