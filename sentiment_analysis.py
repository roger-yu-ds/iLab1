# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:50:10 2019

@author: Roger Yu

This script is used to 
- call the LUIS API to get intents and sentiment scores
- format the sentiment score
- aggregate the sentiment scores
- map to the SERVQUAL topics
- outputs a csv of the sentiment scores and the topics
"""

from UTILS import utils
from UTILS import luis
from UTILS import feature_extraction
from pathlib import Path
import math
import pandas as pd

# Set up paths
output_dir = Path.cwd().joinpath('OUTPUT')
config_dir = Path.cwd().joinpath('CONFIG')
column_dir = output_dir.joinpath('COLUMNS')
luis_output_dir = output_dir.joinpath('LUIS')

# =============================================================================
# Load the data
# =============================================================================

filename = 'student_comments_deduplicated'
student_comments_deduplicated = utils.load_column(column_dir, filename)

# =============================================================================
# Calculate VADER and TextBlob Sentiment Scores
# =============================================================================
df = pd.DataFrame()

for package in ['VADER', 'TextBlob']:
    column_name = f'sentiment_{package.lower()}'
    
    sentiment = feature_extraction.calc_sentiment(
    student_comments=student_comments_deduplicated,
    package=package
    )
    
    df[column_name] = sentiment

# =============================================================================
# Call the LUIS API
# =============================================================================

endpoint = luis.get_luis_url()

batch_size = 50
num_batches = math.ceil(len(student_comments_deduplicated) / batch_size)

# Call the LUIS API
luis.request_api(
        student_comments=student_comments_deduplicated,
        endpoint=endpoint,
        num_batches=num_batches,
        save=True,
        folder=luis_output_dir
        )

# =============================================================================
# Gather the results
# =============================================================================
temp = pd.DataFrame()
for file in luis_output_dir.iterdir():
    print(file.name)
    s = utils.load_column(luis_output_dir, file.name)
    temp = pd.concat([temp, s])

temp.rename(columns={0: 'response'}, inplace=True)
df = df.join(temp)
df = df.join(student_comments_deduplicated)

# =============================================================================
# Extract the topScoringIntent and sentiment score in each response
# =============================================================================
for element in ['sentiment', 'intent']:
    column_name = f'{element}_luis'
    df[column_name] = luis.get_luis_element(
            df.response.apply(lambda x: x.json()),
            element
            )

df.drop(columns=['response'], inplace=True)

# =============================================================================
# Map to SERVQUAL topics
# =============================================================================
servqual_mapping_df = pd.read_csv(config_dir.joinpath('mapping_intents.csv'))

df = df.merge(servqual_mapping_df, on='intent_luis')

# =============================================================================
# Aggregate the sentiments
# =============================================================================
aggregate = feature_extraction.naive_aggregate_sentiments

# axis='columns' will apply the function to each row.
df['sentiment_aggregated'] = df.apply(aggregate, axis='columns')

# =============================================================================
# Merge with all the rows of the data set
# =============================================================================

# The sentiment scoring and intent detection was done only on a subset of the 
# total student comments data, i.e. on the unique lemmatised comments. This 
# sections merges the sentiment scores and the intents with the whole dataset.

student_comments_processed = utils.load_column(
        column_dir,
        'student_comment_processed',
        )

full_df = student_comments_processed.to_frame()

# full_df has the semantically duplicated comments
full_df['student_comment_lemmatised_nostopwords_nopunct'] = (
        student_comments_processed
        .apply(lambda x: ' '.join([token.lemma_ 
                                   for token 
                                   in x 
                                   if not token.is_punct 
                                   and 
                                   not token.is_stop]))
    )
    
full_df = full_df.merge(
        df, 
        left_on='student_comment_lemmatised_nostopwords_nopunct', 
        right_on='student_comments_deduplicated',
        how='left')

# =============================================================================
# Save the columns in column_dir
# =============================================================================

for column in df.columns:
    filename = f'{column}_deduplicated'
    utils.save_object(df[column], column,column_dir)
    
for column in full_df.drop(columns=['student_comment_lemmatised_nostopwords_nopunct',
                                     'student_comments_deduplicated']).columns:
    filename = f'{column}'
    print(filename)
    utils.save_object(df[column], column,column_dir)
   