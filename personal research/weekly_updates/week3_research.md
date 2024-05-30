# Week 2 Research

## Morning Strategy

- Finish reviewing the remaining papers I found that seemed to have relevance. done
- Finalize target locations for the weather API. done
- Obtain weather data. done
- Prepare weather data, crop yield data, and malaria datasets. waiting on weather data but the other 2 are done
- Calculate the similarity scores of the weather data (I might not be able to finish this one, but I hope to end up around here by the end of the day).

## Long-term Strategy

Desired Outcome: Comparative Climate Analysis:

  A. Analyze individual microclimate data streams to compare and calculate similarity scores.
  B. Assess combined microclimate data streams for seasonal variations comparison.
  C. Group years based on microclimate similarities to identify patterns.
  D. Develop predictive models (transformer-based, and Mamba based) linking climate to observed outcomes.
  E. Test models against various seasonal stages to evaluate performance.
  F. Synthesize findings to derive actionable insights.

## TODO

- Read and take notes on:
    - Mamba: A Shallow Dive into a New Architecture for LLMs
    - Impact of Highland Topography Changes on Exposure to Malaria Vectors and Immunity in Western Kenya
    - Long-term Forecasting with TIDE
    - Predicting Malarial Outbreak Using Machine Learning and Deep Learning: A Review

- Finalize target locations for the weather API.
- Obtain weather data.
- Prepare weather data, crop yield data, and malaria datasets.
- Calculate the similarity scores of the weather data.

# Paper Review

## Mamba: A Shallow Dive into a New Architecture for LLMs

- Mamba can process long sequences more efficiently compared to traditional transformers.
- Uses selective state models that dynamically filter and process information based on content, allowing the model to selectively remember or ignore parts of the input.
- Seems good, but it might overfit our small amount of data very quickly.

## Impact of Highland Topography Changes on Exposure to Malaria Vectors and Immunity in Western Kenya

- The pattern of malaria transmission in highland plateau ecosystems is less distinct due to the flat topography and diffuse hydrology resulting from numerous streams.
- The change in land cover on malaria transmission is that deforestation can lead to changes in the microclimate of both adult and larval habitats, hence increasing larvae survival, population density, and gametocytes development in adult mosquitoes.
- Deforestation has been documented to enhance the vectorial capacity of Anopheles gambiae by nearly 100% compared to forested areas.
- Conducting the study in five different ecosystems in the Western Kenyan highlands, they chose 2 V-shaped valleys, two U-shaped valleys, and a plateau.
- Their results indicated that changes in the topography had implications on transmission in the highlands of Western Kenya.
- Plateau and U-shaped valleys were found to have higher parasite density than V-shaped valleys.
- People in the V valleys were less immune than in the plateau and U-valley residents.
- K function was used to determine if spatial distribution of infections in the sites was significantly clustered or if it was random.
- Malaria parasites prevalence varied significantly between the U-shaped and V-shaped valleys.
- The mean parasite prevalence in the U-shaped valleys was 22.1%, in the V-shaped valleys was 2.76%, and at the plateau was 4.42% over the study period.

## Long Term Forecasting with TIDE

- Date: 2023
- Recent work has shown that simple linear models can outperform transformers in certain situations in long-term time series forecasting.
- TIDE is an MLP-based encoder-decoder that is fast and can handle covariates and nonlinear dependencies.

## Predicting Malarial Outbreak Using Machine Learning and Deep Learning: A Review

- Their paper is dated and used old models, doesn't add much to the research but could serve as a good reference when creating our own simpler models.

# Ideas

- I am thinking a model like TIDE will be better than Mamba and a transformer for our task.

# Summary

Today, I went through and reviewed 5 more papers. I found 2 of them to add value for our needs. I don't think for stage 2 that using a transformer from scratch will work; we don't have enough data to gather predictive value from it. However, I think that an interesting approach would be to finetune the ClimaX model on our weather prediction data and see if we can get any predictive value from that. I am interested in looking into how to finetune that model for our needs. I went through and cleaned the output data; we have 26 labelled outputs for crop yield and 108 outputs for malaria outbreak. The data from the malaria database I collected is very old; it ranges only from the '80s-'90s. I don't know if that matters to our project since we do have data from the database that spans those years. When I get home, I will initiate the download of the data we purchased and back it up. For the rest of the day, I continued to search the internet, looking for more papers, most specifically on how to fit small datasets like ours.

# Next Week's Outline For Work

- Prepare the weather data.
- Analyze individual microclimate data streams to compare and calculate similarity scores.
- Assess combined microclimate data streams for seasonal variations comparison.
- Group years based on microclimate similarities to identify patterns.
- Document all of my findings.


