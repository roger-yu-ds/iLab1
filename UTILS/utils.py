# This script is used to load the data

from pathlib import WindowsPath
from pathlib import Path
import pandas as pd
import xlwings as xw
import datetime
import pickle
from typing import Tuple
from typing import Dict
from typing import List
from collections import defaultdict
import spacy
from UTILS import feature_extraction
import numpy as np

def cl_or_wf(filename: str) -> str:
    """Determines if the dataframe is WF or CL"""
    
    if 'CL' in filename.upper():
        result = 'CL'
    elif 'WF' in filename.upper():
        result = 'WF'
    else:
        result = input(f'There is no "CL" or "WF" in the name {filename}, please enter one: ').upper()
    
    return result


def convert_excel_float_to_date(excel_float: float):
    """Fixes the dtype from float to date"""
    
    if excel_float is not None:
        excel_start_date = datetime.datetime(year = 1900, month = 1, day = 1)
        days = datetime.timedelta(days=excel_float)
        adjustment = datetime.timedelta(days = -2)

        result = excel_start_date + days + adjustment
    
    else:
        result = None
    
    return result


def define_dtypes(column: pd.Series, 
                  service: str, 
                  col_dtypes: pd.DataFrame) -> pd.Series:
    
    """Defines the column dtypes of the dataframe"""
    
    col_dtypes = col_dtypes.query('service == @service')
    column_name = column.name
    col_dtypes.set_index(keys = ['service', 'field'], inplace = True)
    col_type = col_dtypes.loc[(service, column_name), 'type']
              
    if col_type is None:
        result = column
    elif col_type == 'datetime64':
        if column.dtype == 'float64':
            result = column.apply(convert_excel_float_to_date)
        else:
            result = column
    elif col_type == 'ordinal':
        result = pd.Categorical(column, ordered = True)
    elif col_type == 'list':
        result = column.str.split(',')
    elif col_type == 'int64':
        result = column.fillna(0).astype(col_type)
    elif col_type == 'str':
        result = column.fillna('').astype(col_type)
    else:
        result = column.astype(col_type)
    
    return result
    

def main(data_path: str, column_dtypes_path: str = None):
    
    df_dict = {}
    for filename in Path(data_path).glob('*.xlsx'):
        print(f'Opening {filename.name}')
        wb = xw.Book(str(filename))
        service = cl_or_wf(filename.name)
        df_dict[service] = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
        
    wb.app.quit()
    
    # Update the column dtypes
    if column_dtypes_path is None:
        column_dtypes_path = Path.cwd().joinpath('CONFIG').joinpath('column_types.csv')
        print(f'Using the default column types {column_dtypes_path}.')
    
    column_dtypes_df = pd.read_csv(column_dtypes_path)
    
    for key, value in df_dict.items():
        df_dict[key] = value.apply(lambda x: define_dtypes(x, key, column_dtypes_df))
        
    
    return df_dict


def save_object(obj, filename: str, output_dir: WindowsPath):
    """Saves the obj as a pickle file in the output_dir as filename
    
    Keyword arguments
    obj -- the object to be saved
    filename -- the name of the file
    output_dir -- a pathlib.WindowsPath object into which the pickle file will 
    be saved
    """
    
    path = output_dir.joinpath(filename)
    
    print(f'Pickling to {path}.')
    
    with open(path, 'wb') as outfile:
        pickle.dump(obj, outfile)
        
#TODO
# Handle existing columns
def add_column(
        df: pd.DataFrame, 
        data_dir: WindowsPath,
        filename: str):
    """Joins a pickled objects (in /OUTPUT/columns) to the right of 
    the DataFrame; join on the index
    
    Keyword arguments
    df -- the dataframe of the student data
    path -- a WindowsPath object 
    filename -- a string containing the name of the file
    """
    df_copy = df.copy(deep=True)

    s = load_column(data_dir, filename)
    # Drop duplicates and keep the first one
    s = s[~s.index.duplicated(keep='last')]
    
    # If the input is a series
    if isinstance(s, pd.core.frame.Series):
        new_cols = [s.name]
    elif isinstance(s, pd.core.frame.DataFrame):
        new_cols = s.columns.tolist()
        
    if all(col in df_copy.columns for col in new_cols):
        print(f'{new_cols} already exists.')
    else:        
        s.name = filename
        
        df_copy = df_copy.join(s)
    
    return df_copy


def load_column(
        path: WindowsPath,
        filename: str):
    """Returns a pickled Series filename path
    
    Keyword arguments
    path -- a WindowsPath object 
    filename -- a string containing the name of the file
    """
    
    with open(path.joinpath(filename), 'rb') as infile:
        s = pickle.load(infile)
    
    s.name = filename
    
    return s


def fix_column_ratings(df: pd.DataFrame):
    """Updates the low student ratings for those with the intent of 
    PRESSED_WRONG_BUTTON, because they intended to rate highly instead, which
    was confirmed by the comments. The mapping 1 → 5, 2 → 4.
    Saves output to disk.
    
    Keyword arguments
    df -- the dataframe of the student data
    """
    
    query = 'luis_intent_pickle == "PRESSED_WRONG_BUTTON" and student_rating < 3'
    df.loc[df.query(query).index, 'student_rating_fixed'] = df.query(query)['student_rating'].map({1:5, 2:4, 3:3})
    
    # Save to disk
    path = Path.cwd().joinpath('OUTPUT').joinpath('student_rating_fixed')
    with open(path, 'wb') as out:
        pickle.dump(df.student_rating_fixed, out)
        
        
def get_seconds_from_timedelta(td):
    """Convert pandas._libs.tslibs.timedeltas.Timedelta to seconds
    
    Keyword arguments
    td -- a pandas._libs.tslibs.timedeltas.Timedelta object
    """
    
    unit = td.days * 24 * 3600 + td.seconds
    return unit


def rename_columns(df: pd.DataFrame, service: str, config_df: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns of the dataframe according to the config file
    
    Keyword arguments
    df -- pd.DataFrame of the data set
    service -- "wf" or "cl"
    config_df -- mapping of original and new column names
    """
    
    service = service.lower()
    
    # Drop the columns that don't have the new name
    config_df = config_df.query('service == @service').dropna(subset = ['new_column_name'], how='any', axis='rows')
    
    # Make a dictionary of {original_column_name: new_column_name}
    rename_dict = {pair[0]:pair[1] for pair in config_df[['original_column_name', 'new_column_name']].to_dict(orient='split')['data']}
    
    return df.rename(columns=rename_dict)


def load_data(
        data_dir: WindowsPath,
        filename: str = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loading the two data sets from xlsx files.
    
    Keyword arguments
    data_dir -- a pathlib.WindowsPath object to the DATA folder
    filename -- a string that contains the location of the data, if None then
        load the original two Excel files
    """
    
    if filename:
        path = data_dir.joinpath(filename)
        
        if '.csv' in filename:
            result = pd.read_csv(path)
        # Else assume it's a pickle file
        else:
            with open(path, 'rb') as infile:
                result = pickle.load(infile)
    else:
        wb = xw.Book(str(data_dir.joinpath('CL_20190823.xlsx')))
        cl_df = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
    
        wb = xw.Book(str(data_dir.joinpath('WF_20190826.xlsx')))
        wf_df = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
        
        wb.app.quit()
        
        result = (cl_df, wf_df)
    
    return result

def load_config(filename: str, config_dir: WindowsPath) -> pd.DataFrame:
    """Load csv config files from the /CONFIG/ folder.
    
    Keyword arguments
    filename -- name of the config file to be loaded, it is assumed to be of 
    csv format
    data_dir -- a pathlib.WindowsPath object to the DATA folder.
    """
    
    filename = filename + '.csv' if '.csv' not in filename else filename
    
    config_path = config_dir.joinpath(filename)
    
    return pd.read_csv(config_path)


def set_column_type(column: pd.Series, 
                    service: str, 
                    config: pd.DataFrame) -> pd.Series:
    """Set the dtype of each column according to the config file.
    
    Keyword arguments
    column -- a column of the data set
    service -- 'CL' or 'WF'
    config -- a dataframe with the mapping of column to data type
    """
    
    service = service.upper()
    
    config = config.query('service == @service')
    column_name = column.name
    col_type = config.loc[(service, column_name), 'type']
              
    if col_type is None:
        result = column
    
    # Somehow loading from Excel leads to a float64 type instead of a date type
    elif col_type == 'datetime64':
        if column.dtype == 'float64':
            result = column.apply(convert_excel_float_to_date)
        else:
            result = column
    elif col_type == 'ordinal':
        result = pd.Categorical(column, ordered = True)
    
    # Some columns, such as the 'qualifications' column contains comma 
    # separated values, these are converted into a list
    elif col_type == 'list':
        result = column.str.split(',')
    elif col_type == 'int64':
        result = column.fillna(0).astype(col_type)
    elif col_type == 'str':
        result = column.fillna('').astype(col_type)
    else:
        result = column.astype(col_type)
    
    return result


def set_column_type2(column: pd.Series, 
                    config: pd.DataFrame) -> pd.Series:
    """Set the dtype of each column according to the config file. Doesn't 
    require the data set to be divided in to "CL" and "WF".
    
    Keyword arguments
    column -- a column of the data set
    service -- 'CL' or 'WF'
    config -- a dataframe with the mapping of column to data type
    """
    
    column_name = column.name
    col_type = config.loc[column_name, 'type']
              
    if (col_type is None) or column_name == 'student_comment_processed':
        result = column
    
    # Somehow loading from Excel leads to a float64 type instead of a date type
    elif col_type == 'datetime64':
        if column.dtype == 'float64':
            result = column.apply(convert_excel_float_to_date)
        else:
            result = column
    elif col_type == 'ordinal':
        result = pd.Categorical(column, ordered = True)
    
    # Some columns, such as the 'qualifications' column contains comma 
    # separated values, these are converted into a list
    elif col_type == 'list':
        result = column.str.split(',')
    elif col_type == 'int64':
        result = column.fillna(0).astype(col_type)
    elif col_type == 'str':
        result = column.fillna('').astype(col_type)
    else:
        result = column.astype(col_type)
    
    return result


def merge_datasets(df_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the CL and WF data sets into one dataframe. Merging is done on 
    the common columns.
    
    Keyword arguments
    df1 -- Either the CL dataframe or the WF dataframe
    df2 -- Either the WF dataframe or the CL dataframe
    """
    
    temp_cl = df_dict['cl'].copy(deep=True)
    temp_wf = df_dict['wf'].copy(deep=True)
    
    # Add the service so they can be grouped later
    temp_cl['service'] = 'cl'
    temp_wf['service'] = 'wf'
    
    cl_set = set(temp_cl.columns)
    wf_set = set(temp_wf.columns)
    
    common_variables = cl_set.intersection(wf_set)
    
    result_df = pd.concat([temp_cl[common_variables], temp_wf[common_variables]], ignore_index=True)
    
    return result_df


def calc_time_difference(s1: pd.Series, s2: pd.Series) -> pd.Series:
    """Calculate the difference in seconds between two columns of type 
    datetime.
    
    Keyword arguments
    s1 -- the series of the earlier time (dtype datetime64), typically the time 
    of submission
    s2 -- the series of the later time (dtype datetime64), typically the time 
    of completion
    """
    
    diff = (s2 - s1).apply(get_seconds_from_timedelta)
    
    return diff


def get_special_characters(comments: pd.Series) -> Dict[str, int]:
    """Finds and counts of non-alphanumeric characters in the series. 
    
    Keyword arguments
    comments -- a spacy.tokens.doc.Doc object
    """
    count_dict = defaultdict(int)
    
    for comment in comments:
        for token in comment:
            if not token.is_alpha and not token.is_punct:
                count_dict[token.orth_] += 1
                
    return count_dict


def replace_with_apostrophe(comments: pd.Series) -> pd.Series:
    """Replaces â€™ with an apostrophe
    
    Keyword arguments
    comments -- a series of dtype 'object'    
    """
    
    comments_replaced = comments.str.replace('â€™', "'")
    return comments_replaced


def remove_stopwords(student_comment: str) -> pd.Series:
    """Remove the stopwords from the student comments
    
    Keyword arguments
    student_comment -- a spacy.tokens.doc.Doc object
    """
    
    return student_comment.apply(lambda x: ' '.join([token.text for token in x if not token.is_stop]))


def get_luis_url(config_path: str = None) -> str:
    """Get the API endpoint for the LUIS.ai API
    
    Keyword arguments
    config_path -- a string to the csv file that contains the authoring key and
        the app ID
    """
    
    if not config_path:
        config_path = (Path
                       .cwd()
                       .parents[0]
                       .joinpath('CONFIG')
                       .joinpath('luis_keys.csv'))
    
    config_df = pd.read_csv(config_path).set_index('key')

    authoring_key = config_df.loc['authoring_key', 'value']
    app_id = config_df.loc['app_id', 'value']
    
    luis_url = f'https://australiaeast.api.cognitive.microsoft.com/luis/v2.0/apps/{app_id}?verbose=true&timezoneOffset=0&subscription-key={authoring_key}'

    return luis_url


def map_columns(
        s: pd.Series,
        mapping_df: pd.DataFrame
        ) -> pd.Series:
    """Map the values of a pd.Series to another one based on the mapping_df. 
    This is used for cleaning up categorical values.
    
    Keyword arguments
    s -- the pd.Series containing the original data
    mapping_df -- the pd.DataFrame containing the mappings
    """
    
    original_df = pd.DataFrame(s)
    
    original_df  = original_df.merge(
            mapping_df,
            how='left',
            left_on=original_df.columns[0],
            right_on=mapping_df.columns[0],
            )
    
    return original_df.iloc[:,-1]


def calc_percentage_counts(
        s: pd.Series,
        names: List[str] = None
        ) -> pd.DataFrame:
    """Create a pd.DataFrame of value counts and percentage counts of a 
    categorical pd.Series
    
    Keyword arguments
    s -- the pd.Series containing the original data
    names -- the column names
    """
    
    value_counts = s.value_counts()
    percentage_counts = value_counts / len(s)
    
    if names:
        col1 = names[0]
        col2 = names[1]
    else:
        col1 = 'count'
        col2 = 'proportion'
    
    df = pd.DataFrame({col1: value_counts,
                       col2: percentage_counts,
                       }
            ).sort_index()
    
    return df


def map_column_dtype(
        column: pd.Series,
        mapping_df: pd.DataFrame,
        ) -> pd.DataFrame:
    """Change the dtype of a pd.Series to the value specified in mapping_df
    
    Keyword arguments
    s -- the pd.Series containing the original data
    names -- the column names
    """
    
    column_copy = column.copy(deep=True)
    
    column_type = mapping_df.loc[column_copy.name, 'type']
    if column_type == 'int64':
        column_copy = column_copy.fillna(0)
        
    column_copy = column_copy.astype(column_type)
    
    return column_copy


def simple_impute(
        df: pd.DataFrame,
        imputation_dict = None
        ) -> pd.DataFrame:
    """Fill the missing data with zeroes and "missing" for numeric and 
    categorical variables, respectively.
    
    Keyword arguments
    s -- the pd.Series containing the original data
    imputation_dict -- the value to impute for a particular data type; the 
        default is {'category': 'missing', 'numeric': 0}
    """
    
    df_copy = df.copy(deep=True)
    if imputation_dict is None:
        imputation_dict = {'category': 'missing',
                           'numeric': 0}
    
    # Categorical variables
    # First add the "missing" category to the categorical variables, then fill
    # the null values with "missing'
    cat_columns = (df_copy
                   .select_dtypes(include='category')
                   .columns
                   )
    
    df_copy.loc[:, cat_columns] = (
            df_copy
            .select_dtypes(include='category')
            .apply(lambda x: x
                   .cat
                   .add_categories(imputation_dict['category'])
                   .fillna('missing')
                   if any(x.isnull()) 
                   else x
                   ) 
            )
    
    # Non-categorical variables
    non_cat_columns = (
            df_copy
            .select_dtypes(exclude='category')
            .columns
            )
    df_copy.loc[:, non_cat_columns] = (
            df_copy
            .select_dtypes(exclude='category')
            .fillna(imputation_dict['numeric'])
            )
    
    return df_copy
    
    
def preprocess(
        df: pd.DataFrame,
        service: str
        ) -> pd.DataFrame:
    """Takes a dataframe of the format CL or WF and outputs a format that is 
    accepted by the machine learning models.
    
    Keyword arguments
    df -- the dataframe containing the rows to be fed into the machine learning
    algorithm for prediction
    service -- either "cl" or "wf"
    """
    
# =============================================================================
#     Set up directories
# =============================================================================
    config_dir = Path.cwd().joinpath('CONFIG')
        
# =============================================================================
#     Define column types
# =============================================================================
    column_types = load_config('mapping_column_types.csv', config_dir)
    column_types.set_index(keys = ['service', 'field'], inplace = True)
    
    df_formatted = df.apply(lambda x: set_column_type(x,
                                                      service.upper(), 
                                                      column_types))
    
# =============================================================================
#     Rename columns
# =============================================================================
    config_column_names = load_config('columns_original.csv', config_dir)
    df_renamed = rename_columns(df_formatted, 
                                service.upper(), 
                                config_column_names)
    
# =============================================================================
#     Add "wait_seconds" to WF
# =============================================================================
    if service.lower() == 'wf':
        s1 = df_renamed.started_at
        s2 = df_renamed.completed_at
        diff = calc_time_difference(s1, s2)
        df_renamed['wait_seconds'] = diff

# =============================================================================
#     Add a service column
# =============================================================================
    df_renamed['service'] = service.lower()
    
# =============================================================================
#     Subset to the common set of columns in both CL and WF
# =============================================================================
    common_columns = pd.read_csv(config_dir.joinpath('common_columns.csv'))
    df_renamed = df_renamed[common_columns.column.values]
    
# =============================================================================
#     Extract features
# =============================================================================
    result_df = df_renamed.copy(deep=True)
    result_df['student_comment_apostrophe'] = (
            replace_with_apostrophe(result_df.student_comment)
            )
    
    # spaCy
    nlp = spacy.load('en_core_web_lg')
    # Create spaCy doc objects
    result_df['student_comment_processed'] = (
            result_df.student_comment_apostrophe.apply(nlp)
            )
    # Remove Stopwords
    result_df['student_comment_no_stopwords'] = (
            remove_stopwords(result_df.student_comment_processed)
            )
    # Cleaning up year_level
    mapping_year_level = load_data(config_dir,'mapping_year_level.csv')
    result_df['year_level_cleaned'] = map_columns(
            result_df.year_level,
            mapping_year_level
            )
    result_df.drop(labels='year_level', axis='columns', inplace=True)
    # Numeric rating, instead of categorical
    result_df['student_rating_numeric'] = (
            result_df.student_rating.astype('float')
            )
    # Length of characters and words
    result_df['student_comment_char_length'] = (
            result_df
            .student_comment
            .apply(lambda x: feature_extraction.get_comment_length(x, 'character'))
            )
    result_df['student_comment_word_length'] = (
            result_df
            .student_comment
            .apply(lambda x: feature_extraction.get_comment_length(x, 'word'))
            )
    # Adding POS tags
    student_comment_pos_tags = (
            feature_extraction
            .get_pos_tags(result_df.student_comment_processed)
            )
    result_df = pd.concat([result_df, student_comment_pos_tags], axis='columns')
    # Adding number of PERSON entities
    result_df['student_comment_num_person_entities'] = (
            result_df
            .student_comment_processed
            .apply(lambda x: feature_extraction.count_entities(x, 'PERSON'))
            )
    
    config_dir = Path.cwd().joinpath('CONFIG')
    
    # Adding tutor and student average ratings
# =============================================================================
#     Load the tutor ratings and student ratings for lookup
# =============================================================================
    tutor_ratings = pd.read_csv(config_dir.joinpath('tutor_ratings.csv'))
    student_ratings = pd.read_csv(config_dir.joinpath('student_ratings.csv'))
    
    # Get the latest score for each tutor_id and student_id, note that  
    # result_df could have multiple rows          
    for elem in [
            'average_tutor_rating_5d_total',
            'average_tutor_rating_over_5d_cl',
            'average_tutor_rating_over_5d_wf',
            ]:
        result_df[elem] = (
            result_df
            .tutor_id
            .apply(lambda x: get_latest_score(df=tutor_ratings,
                                              id_number=x,
                                              student_tutor='tutor',
                                              score_type=elem))
            )
            
    for elem in [
            'average_student_rating_5d_total',
            'average_student_rating_over_5d_cl',
            'average_student_rating_over_5d_wf',
            ]:
        result_df[elem] = (
            result_df
            .student_id
            .apply(lambda x: get_latest_score(df=student_ratings,
                                              id_number=x,
                                              student_tutor='student',
                                              score_type=elem))
            )
        
    # Student start date
    result_df['student_start_date'] = (
            feature_extraction
            .get_student_start_date(result_df)
            )
    
    # Add tutor age
    tutor_dates_df = (result_df[['started_at', 'tutor_birth_year']]
                  .replace(0, np.nan)
                  .dropna(how='any', axis='rows')
                 )
    result_df['tutor_age'] = tutor_dates_df.started_at.dt.year - tutor_dates_df.tutor_birth_year
    
    # Add tutor experience in days
    tutor_days_df = (result_df[['started_at', 'tutor_start_date']]
                         .replace(0, np.nan)
                         .dropna(how='any', axis='rows')
                        )
    result_df['tutor_experience_days'] = (tutor_days_df.started_at - tutor_days_df.tutor_start_date).dt.days
    
    # Add tutor number of sessions
    tutor_sessions = feature_extraction.expanding_count(
        df = result_df,
        grouping_var = 'tutor_id',
        split_by_service = True,
    )
    result_df['tutor_num_sessions_cl'] = tutor_sessions['tutor_num_sessions_cl']
    result_df['tutor_num_sessions_wf'] = tutor_sessions['tutor_num_sessions_wf']
    
    tutor_sessions_total = feature_extraction.expanding_count(
        df = result_df,
        grouping_var = 'tutor_id',
        split_by_service = False,
    )
    result_df['tutor_sessions_total'] = tutor_sessions_total
    
    # Add student number of sessions
    student_sessions = feature_extraction.expanding_count(
        df = result_df,
        grouping_var = 'student_id',
        split_by_service = True,
    )
    result_df['student_num_sessions_cl'] = student_sessions['student_num_sessions_cl']
    result_df['student_num_sessions_wf'] = student_sessions['student_num_sessions_wf']
    
    student_sessions_total = feature_extraction.expanding_count(
        df = result_df,
        grouping_var = 'student_id',
        split_by_service = False,
    )
    result_df['student_sessions_total'] = student_sessions_total    
    
    # Add sex guess
    result_df['sex_guess'] = feature_extraction.guess_sex(result_df.first_name)
    
# =============================================================================
#     Set data types
# =============================================================================
    mapping_column_types_extended = load_data(
            config_dir,
            'mapping_column_types_extended.csv'
            ).set_index('columns')   
    
    result_df['student_rating'] = 0
    
    result_df = (result_df
             .apply(lambda x: map_column_dtype(x, mapping_column_types_extended))
            )  
    
# =============================================================================
#     Fill missing data
# =============================================================================
    result_df = simple_impute(result_df)
    
# =============================================================================
#     Using ml_columns.csv, add empty columns
# =============================================================================
    ml_columns_df = load_data(
            config_dir,
            'ml_columns.csv',
            )
    ml_columns = ml_columns_df.query('use == 1')['columns'].values
    
    missing_cols = set(ml_columns) - set(result_df.columns)
    for col in missing_cols:
        result_df[col] = 0
    
    result_df = result_df[ml_columns]
    
# =============================================================================
#     Define column types
# =============================================================================
    mapping_column_types_extended = load_data(
            config_dir,
            'mapping_column_types_extended.csv'
            ).set_index('columns')
    
    result_df = (result_df
             .apply(lambda x: map_column_dtype(x, mapping_column_types_extended))
            ) 
    
# =============================================================================
#     Get dummies
# =============================================================================
    result_df = pd.get_dummies(result_df,
                            drop_first=True)

    
    
# =============================================================================
#     Convert to dummy variable for the machine learning model
# =============================================================================
    
    return result_df


def get_latest_score(df: pd.DataFrame,
                     id_number: float,
                     student_tutor: str,
                     score_type: str
                     ) -> float:
    """Gets the latest student or tutor score
    """
    score = 3
    if student_tutor == 'tutor':
        score_df = df.query('tutor_id == @id_number')
        score = score_df[score_df.started_at == score_df.started_at.max()][score_type].values[0]
    if student_tutor == 'student':
        score_df = df.query('student_id == @id_number')
        score = score_df[score_df.started_at == score_df.started_at.max()][score_type].values[0]
            
    score = 3 if np.isnan(score) else score
   
    return score























