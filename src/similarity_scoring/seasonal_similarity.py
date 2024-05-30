"""
 * @file seasonal_similarity.py
 * @author Leif Huender
 * @brief 
 * @version 0.1
 * @date 2024-04-19
 * 
 * @copyright Copyright (c) 2024 Leif Huender
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
"""


import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_manipulation.df_filter import DF_Filter
from scipy import stats
import scipy
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

class SeasonalSimilarity:
    def __init__(self, df, numeric_columns=None, string_columns=None):
        self.df = df
        if numeric_columns is None:
            self.numeric_columns = ['lat', 'lon', 'temp', 'dew_point', 'feels_like', 'temp_min', 'temp_max', 'pressure', 'humidity', 'wind_speed', 'wind_deg', 'clouds_all', 'weather_id']
        else:
            self.numeric_columns = numeric_columns

        if string_columns is None:
            self.string_columns = ['weather_main', 'weather_description']
        else:
            self.string_columns = string_columns
       
        self.columns = df.columns
        self.filter = DF_Filter()
        self.calculate = Calculate(self.numeric_columns, self.string_columns)
        self.plot = Plot(self.df)

    def seasonal_similarity(self, start_year, end_year, months):
        df = self.filter.filter(self.df, start_year, end_year, months)
        df['year'] = df['dt'].dt.year
        unique_years = df['year'].unique()
        results = []

        for i, year1 in enumerate(unique_years):
            df1 = df[df['year'] == year1]
            for year2 in unique_years[i+1:]:  # Only comparing each pair once
                df2 = df[df['year'] == year2]
                if df1.shape[0] == df2.shape[0]:
                    for col in self.numeric_columns:
                        df1_col = df1[col].values
                        df2_col = df2[col].values

                        euclid = self.calculate.euclidean(df1_col, df2_col)
                        manhattan = self.calculate.manhattan(df1_col, df2_col)
                        pearson = self.calculate.pearson(df1_col, df2_col)
                        spearman = self.calculate.spearman(df1_col, df2_col)
                        kendall_tau = self.calculate.kendall_tau(df1_col, df2_col)
                        # cosine = self.calculate.cosine(df1_col.reshape(-1,1), df2_col.reshape(-1,1))
                        kl_divergence = self.calculate.kl_divergence(df1_col, df2_col)

                        result = {
                            'Year1': year1, 'Year2': year2, 'Column': col,
                            'Euclidean': euclid, 'Manhattan': manhattan,
                            'Pearson': pearson, 'Spearman': spearman,
                            'Kendall Tau': kendall_tau, #'Cosine': cosine,
                            'KL Divergence': kl_divergence
                        }
                        results.append(result)

        new_df = pd.DataFrame(results)
        return new_df




class Calculate:
    def __init__(self, numeric_columns, string_columns):
        self.numeric_columns = numeric_columns
        self.string_columns = string_columns
    
    def euclidean(self, df1, df2):
        score = scipy.spatial.distance.euclidean(df1, df2)
        return score
    
    def manhattan(self, df1, df2):
        score = scipy.spatial.distance.cityblock(df1, df2)
        return score
    
    def pearson(self, df1, df2):
        score, _ = stats.pearsonr(df1, df2)
        return score
    
    def spearman(self, df1, df2):
        score, _ = stats.spearmanr(df1, df2)
        return score
    
    def kendall_tau(self, df1, df2):
        score = stats.kendalltau(df1, df2)
        return score

    def cosine(self, df1, df2):
        score = cosine_similarity(df1, df2)
        return score
    
    def kl_divergence(self, df1, df2):
        score = scipy.stats.entropy(df1, df2)
        return score

class Plot:
    def __init__(self, df):
        self.df = df
        self.fig = plt.figure()
        self.figs = []

    def heatmap(self, metric, title):
        self.fig = go.Figure(data=go.Heatmap(
            x=self.df['Year1'],  
            y=self.df['Year2'],  
            z=self.df[metric],  
            colorscale='Viridis',  
        ))
        self.fig.update_layout(
            title= title + metric,
            xaxis_title='Year 1',
            yaxis_title='Year 2',
        )
        self.fig.show()

    def save(self, path):
        self.fig.write_html(path)

    def get_fig(self):
        return self.fig
    
    def all(self, metrics, title):
        for metric in metrics:
            self.heatmap(metric, title)
            self.figs.append(self.fig)
    
    def save_all(self, path):
        for i, fig in enumerate(self.figs):
            fig.write_html(path + fig.layout.title.text + '.html')

