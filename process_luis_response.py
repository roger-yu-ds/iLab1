import pickle
from pathlib import Path
from pathlib import WindowsPath
import pandas as pd
import spacy
import argparse
import requests
from UTILS import utils


def get_luis_url(folder: WindowsPath = None) -> str:
    """Create the luis api query url from a csv. Requires the csv to contain
    the endpoint url, app id and primary key
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
        url: str, 
        chunk_size: int = 50
        ) -> pd.Series:
    """Sends student comments to the LUIS.ai API in batches and saves the 
    intemediate results into the OUTPUT folder"""
    
    for i, chunk in enumerate(chunks(student_comments, chunk_size)):
        print(f'Processing batch {i} of size {len(chunk)}')
        
        response = chunk.apply(lambda x: requests.get(f'{url}&q={x}') if x is not None else None)
        response.to_pickle(Path.cwd().joinpath('OUTPUT').joinpath(f'luis_result_{str(i).zfill(4)}'))


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


def main():
    """
    - Load the deduplicated comments
    - Get the luis url
    - Call the API in batches
    - Aggregate the batched pickled files with 'luis_result' in the filename,
    - Extract the top scoring intent, entities and the sentiment of each row
    - Save each of the three series as separate pickle files
    """
    
    data_dir = Path.cwd().joinpath('OUTPUT')
    config_dir = Path.cwd().joinpath('CONFIG')
    
    # Load deduplicated comments
    data = utils.load(data_dir, 'student_comment_deduplicated')
    
    # Get the luis API url
    with open(config_dir.joinpath('luis_url.txt'), 'r') as f:
        luis_url = f.readline()
    
    request_api(
            data,
            luis_url,
            1000,
            )
    

if __name__ == "__main__":
    """
    - Aggregates the batched pickled files with 'luis_result' in the filename,
    - Extracts the top scoring intent, entities and the sentiment of each row
    - Saves each of the three series as separate pickle files
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-lr', action='store', dest='luis_response',
                        help='Enter the luis.ai API json response pickle file'
                        ' name. The default is "luis_response"'
                        )
    
    args = parser.parse_args()
    
    output_dir = Path.cwd().joinpath('OUTPUT')
    
    if args.luis_response is not None:
        luis_response_path = output_dir.joinpath(args.luis_response)
        print('Using the default path: {luis_response_path}.')
    else:
        luis_response_path = Path.cwd().joinpath('OUTPUT').joinpath('luis_response_pickle')
        print(f'Using the default path: {luis_response_path}.')
        
    luis_response = pickle.load(open(luis_response_path, 'rb'))
    if luis_response is not None:
        print('luis_response loaded.')
    
    # Get the top scoring intent, entities and sentiment scores.
    # Store these in the output dir
    for element in ['intent', 'entities', 'sentiment']:
        s = get_luis_element(luis_response, element)
        out_path = output_dir.joinpath(f'luis_{element}_pickle')
        with open(out_path, 'wb') as outfile:
            pickle.dump(s, outfile)