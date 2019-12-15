# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:53:17 2019

@author: Roger Yu

This script contains the functions related to processing of comments using the
LUIS API.
"""

import pickle
from pathlib import Path
from pathlib import WindowsPath
import pandas as pd
import spacy
import requests
import numpy as np


def get_luis_url(folder: WindowsPath = None) -> str:
    """Create the luis api query url from a csv. Requires the csv to contain
    the endpoint url, app id and primary key. These can be obtained from the
    luis.ai site:
        https://www.luis.ai/applications/{app_ic}/versions/0.1/manage/endpoints
    """
    if folder is None:
        folder = Path.cwd().joinpath('CONFIG')
    
    path = folder.joinpath('luis_keys.csv')
    
    df = pd.read_csv(path, index_col='key')
    endpoint = df.loc['endpoint', 'value']
    app_id = df.loc['app_id', 'value']
    primary_key = df.loc['subscription_key', 'value']
    
    result = (
            f'{endpoint}luis/v2.0/apps/{app_id}?verbose=true&timezoneOffset'
            f'=0&subscription-key={primary_key}&q='
            )
    
    return result


def drop_semantic_duplicates(
        student_comments: pd.Series
        ) -> pd.Series:
    """Remove the duplicate comments based on the comment's semantics, e.g.
    "I was waiting for 1 hour." and "I waited for 1 hour." are orthographically
    different, but the semantically the same, so one will be removed
    
    Keyword arguments:
    student_comments -- a pd.Series of spacy.tokens.doc.Doc objects
    """
    
    # Remove the stopwords and remove punctuation
    student_comment_lemmatised_nostopwords_nopunct = (
            student_comments
            .apply(lambda x: ' '.join(
                    [token.lemma_ 
                     for token 
                     in x 
                     if not token.is_punct 
                     and not token.is_stop]
                    )
                   )
    )
    
    student_comment_lemmatised_nostopwords_nopunct = (
            student_comment_lemmatised_nostopwords_nopunct
            .drop_duplicates(keep='first')
            )
            
    return student_comment_lemmatised_nostopwords_nopunct


def request_api(
        student_comments: pd.Series, 
        endpoint: str, 
        num_batches: int = 50,
        save: bool = True,
        folder: WindowsPath = None
        ) -> pd.Series:
    """Sends student comments to the LUIS.ai API in batches and saves the 
    intemediate results into the OUTPUT folder
    
    Keyword arguments
    student_comments -- the pd.Series that contains the student comments in 
        natural language
    endpoint -- the luis endpoint
    num_batches -- the number of batches into which the comments would be 
        grouped and sent to the api
    save -- a boolean that saves the raw json response to local disk
    folder -- a location to which to save the response data, defaults to 
        /OUTPUT/LUIS/
    """
    if save and folder is None:
        folder = Path.cwd().joinpath('OUTPUT').joinpath('LUIS')
    
    for i, batch in enumerate(np.array_split(student_comments, num_batches)):
        print(f'Processing batch {i} of {num_batches}:')
        
        luis_result = batch.apply(lambda x: requests.get(f'{endpoint}{x}'))

        # Saving the results to pickle
        filename = f'luis_result_{str(i).zfill(4)}'
        luis_result.to_pickle(folder.joinpath(filename))
        
        print(f'Saved to {folder.joinpath(filename)}.')
        

def chunks(array, size: int):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(array), size):
        yield array[i:i + size]
        

def load_pickles(directory: str) -> pd.Series:
    """Load the batch response from calling the LUIS.ai API. The files are in
    pickle format.
    
    keyword arguments
    directory -- the directory that contains the pickle files
    """
    concat_series = pd.Series()
    for path in Path(directory).glob('luis_result_[0-9][0-9][0-9][0-9]*'):
        luis_series = pickle.load(open(path, 'rb'))
        concat_series = pd.concat([concat_series, luis_series])
    
    return concat_series.apply(lambda x: x.json() if x.status_code == 200 else None)

def merge_comments(df1: pd.DataFrame, df2: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Merging the results from the luis response with the lemmatised comments.
    
    Fix index: for some reason the index of the pd.Series on which the 
    luis.api call was applied is not consistent. The top contains integer 
    index, while the bottom contains datetime index, which is the start_at
    column.
    
    The correct index is supposed to be the integer one. 
     
    The unique student comments with apostrophe was saved on file as
    student_comments_apostrophe.csv with the integer index. The index
    of the luis_result series will be fixed by merging the luis_result 
    pd.Series with the student_comments_apostrophe series. The join 
    key will be the "query" element of the luis.ai api json response.    
    """

    nlp = spacy.load('en_core_web_lg')
    
    df1 = pd.DataFrame(load_pickles(out_dir))
    df1.columns = ['response']
    df1['student_comment_apostrophe'] = df1.response.apply(lambda x: x['query'] if x is not None else None)    
    df1.dropna(subset=['student_comment_apostrophe'], inplace=True)
    
    df2 = pd.read_csv("D:/OneDrive - UTS/36102 iLab 1 - Spring 2019/CODE/OUTPUT/student_comment_lemmatised_nostopwords_nopunct.csv")   
    df2.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    
    # Create a merging column by putting the comment through the nlp 
    # preprocessing steps
    df1['merge_col'] = df1.student_comment_apostrophe.apply(lambda x: ' '.join([token.lemma_ for token in nlp(x) if not token.is_punct and not token.is_stop]))
    
    merged_df = df2.merge(df1,
                          how='left',
                          left_on='student_comment_lemmatised_nostopwords_nopunct',
                          right_on='merge_col')
    merged_df.dropna(inplace=True)
    merged_df.set_index('index', inplace=True)    
    
    return merged_df


def get_luis_element(s: pd.Series, element: str) -> pd.Series:
    """Extract the element, e.g. topScoringIntent, from the LUIS.api json 
    response
    
    s -- the pd.Series of the unravelled json response
    element -- the elements, such as "sentiment" or "intent"
    """
    result = 0
    if element == 'intent':
        result = s.apply(lambda x: x['topScoringIntent']['intent'])
    elif element == 'entities':
        result = s.apply(lambda x: x['entities'])
    elif element == 'sentiment':
        valence = s.apply(lambda x: x['sentimentAnalysis']['label']).map({'positive': 1,
                                                                         'neutral': 0,
                                                                         'negative': -1})
        result = s.apply(lambda x: x['sentimentAnalysis']['score']) * valence
    
    return result