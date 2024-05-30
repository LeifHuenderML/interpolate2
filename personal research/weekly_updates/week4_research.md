# Week 4 Research 

## Morning Strategy

- Go through and add labels for the missing location labels for the weather data
- Analyze individual microclimate data streams to compare and calculate similarity scores.
- Assess combined microclimate data streams for seasonal variations comparison.
- Group years based on microclimate similarities to identify patterns.
- Document all of my findings.


# Summary 
Today, I added all the missing labels for the weather dataset. I had to align the timestamps for the individual microclimates so that when we compare similarity scores, we will obtain accurate metrics. This task took a while as I was struggling to find a way to align them, and it turned out the issue was that there were duplicates in the datasets causing some entries to be overlooked. Once I resolved this, the rest of the data cleaning went smoothly. The complete dataset contains more than 3 million rows, which will be valuable for gleaning information across various parameters. I began writing the tools to calculate similarity scores across each of the columns. I created a class with different functions, such as calculate_euclidean_similarity, which will calculate a similarity score for each of the columns between the 4 microclimates. I haven't gotten to thoroughly test it to ensure it is providing accurate metrics. That is what I plan to do next week. So far, I have created ways to get scores for Euclidean similarity, Manhattan similarity, Pearson similarity, Spearman similarity, Kendall tau similarity, and cosine similarity. Each of the scores is written into a dictionary that is easy to parse and obtain relevant information from. I plan on compiling all the scores from the microclimates and turning them into heatmaps to visualize the similarities between various parameters and microclimates. Next week, I will continue to finish my testing of the accuracy of these metrics to see if they are finding useful information. I also plan to split the individual microclimates by seasons and then run similarity metrics for individual microclimates by each year. Please let me know if you have any questions or comments.
