import pandas as pd
from collections import Counter
import gender_guesser.detector as gender
import re
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def rolling_mean(df: pd.DataFrame, 
                 agg_col: str, 
                 grouping_var: str,
                 window,
                 split_by_service,
                 sort_by: str = 'started_at') -> pd.Series:
    """Calculates the rolling mean of agg_col grouped by the group. The time 
    horizon excludes the current row, this is because the current row's rating
    is unknown at the `started_at` time, i.e. the rating for that particular
    row is only given at the end of the session.
    
    Keyword arguments
    df -- the main dataframe with the data
    agg_col -- the agg_column whose values will be aggregated
    grouping_var -- the grouping variable, e.g. 'student_id' or 'tutor_id'
    window -- either an integer for the previous number of entries in the group,
    or an alias such as '30d' for the last 30 days
    sort_by -- the column on which to sort
    """
    
    temp_df = df.copy(deep=True)
    series_name = f'average_rating_over_{window}'
    service_var = 'service'
    
    if split_by_service:
        grouping_vars = [service_var, grouping_var]
        subset_vars = [service_var, grouping_var, sort_by, agg_col]
    else:
        grouping_vars = [grouping_var]
        subset_vars = [grouping_var, sort_by, agg_col]
    
    # Subset the df
    ts = df[subset_vars]
    ts = ts.sort_values(by=[grouping_var, sort_by])
    
    # Subset just the rows with student_rating
    ts = ts.query('student_rating > 0')
    ts = ts.set_index('started_at')
    
    # Calculate the rolling mean
    average_rating = (ts
            .groupby(grouping_vars)[agg_col]
            .rolling(window, closed='left')
            .mean()
            .fillna(method='bfill')
            )
    
    average_rating.name = series_name
    
    # Merge with all the rows, i.e. including those that don't have 
    # student_rating and backfill the blanks
    average_rating_df = average_rating.reset_index()
    average_rating_df.rename(
            columns={agg_col: series_name},
            inplace = True,
            )
    
    # Merge with df to get the original indexes
    if split_by_service:
        results = {}
        for service in ['cl', 'wf']:
            average_rating_df_subset = (average_rating_df
                                        .query('service == @service')
                                        .drop(labels='service', axis = 'columns')
                                        )
            average_rating_full_df = df.merge(average_rating_df_subset, 
                                       how='left',
                                       on=[grouping_var, sort_by]
                                       )[[grouping_var, sort_by, series_name]]
            average_rating_full = (average_rating_full_df
                                      .sort_values(by=[grouping_var, sort_by])
                                      .groupby(grouping_var)
                                      .fillna(method='ffill')
                                      )[series_name]
            results[f'{series_name}_{service}'] = average_rating_full
    else:
        average_rating_full_df = df.merge(average_rating_df, 
                                       how='left',
                                       on=[grouping_var, sort_by]
                                       )[[grouping_var, 
                                          sort_by,
                                          series_name]]
    
        # For rows that don't have a rating, fill the average ratings for the rows
        # that don't have ratings
        results = (average_rating_full_df
                  .sort_values(by=[grouping_var, 'started_at'])
                  .groupby(grouping_var)
                  .fillna(method='ffill')
                  )[series_name]
    
    return results
    

def get_comment_length(student_comment: str, len_type: str) -> int:
    """Get the character or word length of the student comment
    
    Keyword arguments
    student_comment -- a string containing the student comment
    len_type -- either 'character' or 'word'
    """
    
    if len_type == 'word':
        result = len(student_comment.split(' ')) if len(student_comment) > 0 else 0
    elif len_type == 'character':
        result = len(student_comment)
        
    return result

def get_pos_tags(student_comment: str) -> pd.DataFrame:
    """Get the POS (part of speech) tags for each of the words in the student
    comments
    
    Keyword arguments
    student_comment -- a spacy.tokens.doc.Doc object
    """
    
    # Count how many of each pos tags are in each comment
    pos_tags = student_comment.apply(lambda x: Counter([token.pos_ for token in x]))
    
    # Expand the list column into several columns
    pos_tags_df = pos_tags.apply(pd.Series).fillna(0)
    
    return pos_tags_df


def count_entities(doc, ent_label):
    """Counts the number of entities in doc that is of type ent_label
    
    Keyword arguments
    doc -- a spacy.tokens.doc.Doc object
    ent_label -- the entity lable to be checked, e.g. "PERSON"
    """
    
    return len([ent for ent in doc.ents if ent.label_ == 'PERSON'])


def guess_sex(s: pd.Series):
    """Guess the sex of the student given their first name(s). The name must
    be in title format for the gender_guesser to work. The "mostly_" versions
    have been mapped to their non-mostly counterparts for simplicity. "andy"
    (androgynous), i.e. equally male and female, will be mapped to "unknown".
    
    Keyword arguments
    name -- a string that contains the first name
    """
    
    gender_detector = gender.Detector()
    
    # Compound names cannot be detected. Split the compound names and use the 
    # first part only.
    first_word = (s
                  .apply(lambda x: re.split(' |-', x)[0].title())
                  )
    
    sex_mapping = {'male': 'male'
                   ,'female': 'female'
                   ,'mostly_female': 'female'
                   ,'mostly_male': 'male'
                   ,'andy': 'unknown'
                   ,'unknown': 'unknown'}
    
    sex_guess = (first_word
                 .apply(gender_detector.get_gender)
                 .map(sex_mapping)
                 )
    
    return sex_guess


def expanding_count(df: pd.DataFrame, 
                    grouping_var: str,
                    split_by_service: bool = False,
                    sort_by: str = 'started_at'
                   ) -> pd.Series:
    """Calculates the expanding count of col grouped by the group (typically by
    student_id or tutor_id before the session, i.e. how much experience the 
    student has had with the service in terms of the number of previous 
    sessions
    
    The time horizon starts from the date that the person joined.
    
    Keyword arguments
    df -- the main dataframe with the data
    grouping_var -- the grouping variable, 'student_id' or 'tutor_id'
    split_by_sevice -- boolean to calculate the number of sessions separately 
        for CL and WF
    sort_by -- the column on which to sort
    """
    temp_df = df.copy(deep=True)
    series_name = f'{grouping_var.split("_")[0]}_num_sessions'
    service_var = 'service'
    
    if split_by_service:
        grouping = [service_var, grouping_var]
        subset_vars = [service_var, grouping_var, sort_by]
    else:
        grouping = [grouping_var]
        subset_vars = [grouping_var, sort_by]
        
    # Subset the df
    ts = df[subset_vars]
    ts = ts.sort_values(by=subset_vars)
    ts = ts.set_index(sort_by)
    
    # Calculate the expanding count from the beginning. Note that the -1 is 
    # because we are not counting the current session.
    num_sessions = (ts
            .groupby(grouping)
            .expanding()
            .count()
            ) - 1
    
    # Need only one column since their just counts
    num_sessions = num_sessions.iloc[:,0]
    num_sessions.name = series_name
    num_sessions = num_sessions.reset_index()
    
    # Merge back with the main df to separate the CL and WF
    if split_by_service:
        results = {}
        for service in ['cl', 'wf']:
            num_session_df = (num_sessions
                                 .query('service == @service')
                                 .drop(labels='service', axis='columns')
                                 )
            #TODO
            # The merging for CL and WF don't result in the same number of rows
            # cl: 506287, wf: 506373. This could be due to the same combination
            # of the values in the joining keys. It is known that the datetime
            # is not unique.
            num_session_cl = (temp_df
                              .merge(num_session_df, 
                                     how='left',
                                     on=[grouping_var, sort_by])
                              )[series_name]
            
            series_name_service = f'{series_name}_{service}'
            num_session_cl.name = series_name_service
            results[series_name_service] = num_session_cl
    else:
        
        num_session_merged = (temp_df
                              .merge(num_sessions, 
                                     how='left',
                                     on=[grouping_var, sort_by])
                              )[series_name]
    
        series_name_service = f'{series_name}_total'
        num_session_merged.name = series_name_service
        results = num_session_merged

    return results


def get_student_start_date(df: pd.DataFrame) -> pd.Series:
    """Calculate the ealiest date for each student_id.
    
    Keyword arguments
    df -- main df
    """
    
    start_date = df.groupby('student_id')['started_at'].min()
    start_date.name = 'student_start_date'
    
    # Merge back to the main df to get a value for all the rows
    student_start_date_full = df.merge(
            start_date,
            how='left',
            on='student_id')['student_start_date']
    
    return student_start_date_full


def calc_sentiment_textblob(text: str):
    """Calculate sentiment score of a comment
    
    Keyword arguments
    text -- a string that contains the unprocessed comment
    """
    
    text = str(text)
    blob = TextBlob(text)
    
    return blob.sentiment.polarity


#IMPROVEMENTS
# Finish the LUIS one
def calc_sentiment(
        student_comments: pd.Series,
        package: str
        ) -> pd.Series:
    """Calculate sentiment scores based on the chosen package
    
    Keyword arguments
    student_comments -- a pd.Series containing the unprocessed comments
    package -- either 'VADER', 'TextBlob', or 'LUIS'
    """
    
    if package.lower() == 'vader':
        sid = SentimentIntensityAnalyzer()
        results = student_comments.apply(
                lambda x: sid.polarity_scores(x)['compound']
                )
        results.name = 'sentiment_vader'
    elif package.lower() == 'textblob':
        results = student_comments.apply(calc_sentiment_textblob)
        results.name = 'sentiment_textblob'
    elif package.lower() == 'luis':
        pass
        
    return results


#IMPROVEMENTS
#Include an arbitrary number of columns
def naive_aggregate_sentiments(row):
    """Take the average of the two closest values of the three sentiment scores
    calculate by TextBlob, VADER (nltk) and LUIS.
    
    Keyword arguments:
    row -- a pd.Series that is a row of a pd.DataFrame
    """
    sentiment_list = sorted(row[['sentiment_textblob', 'sentiment_vader', 'sentiment_luis']])
    if any(np.isnan(sentiment_list)):
        result = np.nanmean(sentiment_list)
    elif sentiment_list[1] - sentiment_list[0] < sentiment_list[2] - sentiment_list[1]:
        result = np.nanmean(sentiment_list[:2])
    else:
        result = np.nanmean(sentiment_list[1:])
    return result