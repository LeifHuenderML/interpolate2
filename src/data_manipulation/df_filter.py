"""
 * @file df_slicer.py
 * @author Leif Huender
 * @brief allows for you to slice a dataframe by variational dates, 
          for instance with trying to measure pecan yield the most important 
          parts of the weather data are the monthes during the growing season.
          this slicer allows you to specify the range of the dates you want and 
          will break it up over the years and return the data.
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
import pandas as pd

class DF_Filter:

    # filter the data by the start and end year and the months you want to keep
    def filter(self, df, start_year, end_year, months):
        # check if the date column is a datetime object
        if not pd.api.types.is_datetime64_any_dtype(df['dt']):
            df['dt'] = pd.to_datetime(df['dt'])
        # filter the data
        mask = (df['dt'].dt.year >= start_year) & (df['dt'].dt.year <= end_year) & (df['dt'].dt.month.isin(months))
        df = df.loc[mask]
        return df
    
    # save the data to a csv file
    def save(self, df, path):        
        df.to_csv(path, index=False)
