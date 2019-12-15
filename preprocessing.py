#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# This notebook achieves these two goals:
# - formatting the original data into a consistent format
# - feature engineering, i.e. creating new columns from the existing columns, which are saved in the `/OUTPUT/COLUMNS`directory

# In[ ]:


import luis #https://pypi.org/project/luis/

import requests
from pathlib import Path

import spacy
from spacy import displacy
from spacy.lang.en import English
from spacy.matcher import Matcher

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob

import pandas as pd
import pandas_profiling as pp
import featuretools as ft
import numpy as np
from scipy import stats

import datetime

from collections import Counter
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import getpass

import xlwings as xw

import os

#import gender_detector
import gender_guesser.detector as gender

from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

import pickle

import re

from itertools import product

from UTILS import utils
from UTILS import feature_extraction
import process_luis_response

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('vader_lexicon') # VADER (Valence Aware Dictionary and sEntiment Reasoner, https://github.com/cjhutto/vaderSentiment)
# In[ ]:


pd.set_option('display.max_colwidth', 1000)
pd.options.display.max_columns = None


# In[ ]:


data_dir = Path.cwd().joinpath('DATA')
#output_dir = Path.cwd().joinpath('OUTPUT')
output_dir = Path.cwd().joinpath('OUTPUT').joinpath('TEST')
column_dir = output_dir.joinpath('COLUMNS')
config_dir = Path.cwd().joinpath('CONFIG')
image_path = output_dir.joinpath('IMAGES')


# # Load the Data<a id='load-data'></a>

# In[ ]:


cl_df, wf_df = utils.load_data(data_dir)


# ## Load the Config File

# The file contains the data dictionary; the data types are in the pandas format.

# In[ ]:


config_dir = Path.cwd().joinpath('CONFIG')


# In[ ]:


column_types = utils.load_config('mapping_column_types.csv', config_dir)
column_types.set_index(keys = ['service', 'field'], inplace = True)
column_types.head()


# # Format the Columns<a id='format-columns'></a>

# Create a functions to set the correct `dtype` (datatype) for the columns.

# In[ ]:


cl_df_formatted = cl_df.apply(lambda x: utils.set_column_type(x, 'CL', column_types))
wf_df_formatted = wf_df.apply(lambda x: utils.set_column_type(x, 'WF', column_types))


# # Pandas Profiling<a id='pandas-profiling'></a>

# Pandas profiling shows basic statistics of each column in the data set. This is useful in getting to quickly get a high-level feel of the data.

# In[ ]:


cl_profile = pp.ProfileReport(cl_df_formatted)


# In[ ]:


wf_profile = pp.ProfileReport(wf_df_formatted)


# ## Connect Live Profile

# In[ ]:


cl_profile


# ## Writing Feedback Profile

# In[ ]:


wf_profile


# ## Change Variable Names

# The `start_at_utc` variable in the CL service is very similar to the `submitted_at` variable in the WF service. They're both the times at which the student requests an interaction with a specialist, see `/CONFIG/mapping_column_types.csv` for details.
# 
# Both will be changed to `start_at`. This step uses `/CONFIG/columns_original.csv`, which was specified manually.

# In[ ]:


config_column_names = utils.load_config('columns_original.csv', config_dir)


# In[ ]:


cl_df_renamed = utils.rename_columns(cl_df_formatted, 'CL', config_column_names)
wf_df_renamed = utils.rename_columns(wf_df_formatted, 'WF', config_column_names)


# # Data Preprocessing

# ## Add/calculate the `wait_seconds` for WF

# The `wait_seconds` column for the WF does not exist in the original data. However, this can be calculated as the difference between the `completed_at` and `started_at` (previously `submitted_at`) columns.

# In[ ]:


s1 = wf_df_renamed.started_at
s2 = wf_df_renamed.completed_at
diff = utils.calc_time_difference(s1, s2)


# In[ ]:


wf_df_renamed['wait_seconds'] = diff


# In[ ]:


wf_df_renamed.head()[['started_at', 'completed_at', 'wait_seconds']]


# ## Combine the Data Sets

# The CL and WF data sets are merged to form one large data set of around 500,000 rows.

# In[ ]:


df_merged = utils.merge_datasets(
    {'cl': cl_df_renamed, 
     'wf':wf_df_renamed,
    }
)


# In[ ]:


df_merged.shape


# There are only 21 columns that overlap in both data sets.

# In[ ]:


df_merged.columns


# In[ ]:


utils.save_object(df_merged, 'df_merged', output_dir)


# # Text Preprocessing

# Subset the data to just those with comments. Subsequent processes on comments should be done on just the rows that have comments.

# In[ ]:


length_word = (df_merged
               .student_comment
               .apply(lambda x: 
                      feature_extraction
                      .get_comment_length(x,
                                          len_type='word'
                                         )
                     )
              )


# In[ ]:


mask_has_comment = length_word > 0


# In[ ]:


index_has_comment = df_merged[mask_has_comment].index


# ## Replacing `â€™` with `'`

# There are some strange characters such as ™ and â€. It turns out that â€™ is actually an apostrophe. The source file seems to have had an error in the extraction process. So the next step is to substitute `â€™` with `'`.

# In[ ]:


df_merged.student_comment[df_merged.student_comment.str.contains('â€™')][:5]


# In[ ]:


student_comment_apostrophe = utils.replace_with_apostrophe(df_merged.student_comment[mask_has_comment])
utils.save_object(student_comment_apostrophe, 'student_comment_apostrophe', column_dir)


# In[ ]:


student_comment_apostrophe[df_merged.student_comment.str.contains('â€™')][:5]


# ## Creating Spacy `Doc` Objects

# Loading a `spaCy` pretrained statistical model, see the [documentation](https://spacy.io/models/en#en_core_web_lg). The model processes each student comment and attaches part of speech tags such as NOUN, VERB, etc. It takes about 8 minutes to process all the available comments.

# In[ ]:


nlp = spacy.load('en_core_web_lg')


# In[ ]:


start_time = datetime.datetime.now()
student_comment_processed = student_comment_apostrophe[mask_has_comment].apply(nlp)
end_time = datetime.datetime.now()

print(f'Tokenising took {end_time - start_time}.')


# In[ ]:


utils.save_object(student_comment_processed, 'student_comment_processed', column_dir)


# ## Remove Stopwords

# Stop words are words such as 'to', 'from', 'the', 'a'. These do not carry meaning by themselves. It's common to remove them prior to further processing (see [documentation](https://spacy.io/usage/adding-languages#stop-words).

# In[ ]:


from spacy.lang.en.stop_words import STOP_WORDS


# In[ ]:


student_comment_no_stopwords = utils.remove_stopwords(student_comment_processed)


# In[ ]:


student_comment_no_stopwords.head()


# In[ ]:


utils.save_object(student_comment_no_stopwords, 'student_comment_no_stopwords', column_dir)


# ## Cleaning Up `year_level`

# The values in these columns have been mapped according to the files in `/CONFIG/mapping_year_level.csv`. This is to ensure that several similar values are mapped to the same category, making subsequent machine learning algorithms to perform better.

# In[ ]:


mapping_year_level = utils.load_data(
    config_dir,
    'mapping_year_level.csv')


# In[ ]:


year_level_cleaned = utils.map_columns(
    df_merged.year_level,
    mapping_year_level
    )


# In[ ]:


year_level_cleaned.head()


# In[ ]:


utils.save_object(
    year_level_cleaned,
    year_level_cleaned.name,
    column_dir
)


# # Feature Engineering

# ## Create a Numerical Version of the `student_rating`

# Some models require the dependent variable to be numeric. Furthermore, the EDA (exploratory data analysis) will numerically aggregate this column, e.g. average rating for a particular student or tutor.

# In[ ]:


student_rating_numeric = df_merged.student_rating.astype('float')
student_rating_numeric.name = 'student_rating_numeric'


# In[ ]:


utils.save_object(
    obj=student_rating_numeric, 
    filename=student_rating_numeric.name, 
    output_dir=column_dir
    )


# ## Length of the `student_comment`

# In[ ]:


length_char = student_comment_apostrophe.apply(lambda x: feature_extraction.get_comment_length(x, 'character'))
length_word = student_comment_apostrophe.apply(lambda x: feature_extraction.get_comment_length(x, 'word'))


# In[ ]:


length_char.name = 'student_comment_char_length'
length_word.name = 'student_comment_word_length'


# In[ ]:


utils.save_object(length_char, length_char.name, column_dir)
utils.save_object(length_word, length_word.name, column_dir)


# In[ ]:


sns.distplot(length_word, kde=False).set_title('Distribution of word length including blanks')


# In[ ]:


length_word[length_word > 0].describe()


# In[ ]:


student_comment_apostrophe[length_word == 1]


# ## Adding POS (part of speech) Tags

# POS (part of speech) include NOUN, VERB, and ADJECTIVE, see [Part-of-speech tagging](https://spacy.io/api/annotation#pos-tagging).

# In[ ]:


student_comment_pos_tags = feature_extraction.get_pos_tags(student_comment_processed[mask_has_comment])
student_comment_pos_tags.name = 'student_comment_pos_tags'


# In[ ]:


utils.save_object(student_comment_pos_tags, student_comment_pos_tags.name, column_dir)


# In[ ]:


student_comment_processed.head()


# In[ ]:


for token in student_comment_processed.iloc[0]:
    print(token.text, token.pos_)


# In[ ]:


student_comment_pos_tags.head()


# ## Number of `PERSON` Entities

# In[ ]:


student_comment_num_person_entities = student_comment_processed.apply(lambda x: feature_extraction.count_entities(x, 'PERSON'))
student_comment_num_person_entities.name = 'student_comment_num_person_entities'


# In[ ]:


utils.save_object(student_comment_num_person_entities, student_comment_num_person_entities.name, column_dir)


# In[ ]:


student_comment_num_person_entities.head()


# In[ ]:


for ent in student_comment_processed[17].ents:
    print({ent.text, ent.label_})


# In[ ]:


for ent in student_comment_processed[27].ents:
    print({ent.text, ent.label_})


# ## Average Score of Tutors (combined by CL and WF)

# In[ ]:


col = 'student_rating'
grouping_var = 'tutor_id'
window = '5d'
split_by_service=False

series_name = f'average_tutor_rating_{window}_total'
print(series_name)


# In[ ]:


average_tutor_rating_5d_total = feature_extraction.rolling_mean(
    df = df_merged,
    agg_col = 'student_rating',
    grouping_var = grouping_var,
    window = window,
    split_by_service=False,
)


# In[ ]:


average_tutor_rating_5d_total.name = series_name


# In[ ]:


average_tutor_rating_5d_total[average_tutor_rating_5d_total.notnull()].head()


# In[ ]:


utils.save_object(average_tutor_rating_5d_total, 
                  series_name, 
                  column_dir)


# ## Average Score of Tutors (split by CL and WF)

# The average score of the tutors is calculated for a given time horizon (5 days) up to the datetime of the session.

# In[ ]:


col = 'student_rating'
grouping_var = 'tutor_id'
window = '5d'
split_by_service=True


# In[ ]:


tutor_average_score = feature_extraction.rolling_mean(
    df = df_merged,
    agg_col = 'student_rating',
    grouping_var = grouping_var,
    window = window,
    split_by_service=True,
)


# In[ ]:


average_tutor_rating_over_5d_cl = tutor_average_score['average_rating_over_5d_cl']
average_tutor_rating_over_5d_wf = tutor_average_score['average_rating_over_5d_wf']


# In[ ]:


average_tutor_rating_over_5d_cl.name = 'average_tutor_rating_over_5d_cl'
average_tutor_rating_over_5d_wf.name = 'average_tutor_rating_over_5d_wf'


# In[ ]:


utils.save_object(average_tutor_rating_over_5d_cl,
                  'average_tutor_rating_over_5d_cl',
                  column_dir)

utils.save_object(average_tutor_rating_over_5d_wf,
                  'average_tutor_rating_over_5d_wf',
                  column_dir)


# ## Average Score of Students (combined CL and WF)

# The average score of the tutors is calculated for a given time horizon (5 days) up to the datetime of the session.

# In[ ]:


col = 'student_rating'
grouping_var = 'student_id'
window = '5d'

series_name = f'average_student_rating_{window}_total'
print(series_name)


# In[ ]:


average_student_rating_5d_total = feature_extraction.rolling_mean(
    df=df_merged,
    agg_col='student_rating',
    grouping_var=grouping_var,
    window='5d',
    split_by_service=False
)


# In[ ]:


average_student_rating_5d_total.name = series_name


# In[ ]:


average_student_rating_5d_total[average_student_rating_5d_total.notnull()].head()


# In[ ]:


utils.save_object(average_student_rating_5d_total, 
                  series_name, 
                  column_dir)


# ## Average Score of Students (split by CL and WF)

# In[ ]:


col = 'student_rating'
grouping_var = 'student_id'
window = '5d'
split_by_service=True


# In[ ]:


tutor_average_score = feature_extraction.rolling_mean(
    df = df_merged,
    agg_col = 'student_rating',
    grouping_var = grouping_var,
    window = window,
    split_by_service=True,
)


# In[ ]:


average_student_rating_over_5d_cl = tutor_average_score['average_rating_over_5d_cl']
average_student_rating_over_5d_wf = tutor_average_score['average_rating_over_5d_wf']


# In[ ]:


average_student_rating_over_5d_cl.name = 'average_student_rating_over_5d_cl'
average_student_rating_over_5d_wf.name = 'average_student_rating_over_5d_wf'


# In[ ]:


utils.save_object(average_rating_over_5d_cl,
                  'average_student_rating_over_5d_cl',
                  column_dir)

utils.save_object(average_rating_over_5d_wf,
                  'average_student_rating_over_5d_wf',
                  column_dir)


# Save a dataframe of `tutor_id`, `started_at` and the various scores. This is to lookup the score of a particular `tutor_id` for the prediction set.

# In[ ]:


type(tutor_average_score)


# In[ ]:


(df_merged
 .join(average_tutor_rating_5d_total)
 .join(average_tutor_rating_over_5d_cl)
 .join(average_tutor_rating_over_5d_wf)[['tutor_id', 
                                         'started_at', 
                                         'average_tutor_rating_5d_total',
                                         'average_tutor_rating_over_5d_cl',
                                         'average_tutor_rating_over_5d_wf']]
 .to_csv(config_dir.joinpath(f'tutor_ratings.csv'))
)


# In[ ]:


(df_merged
 .join(average_student_rating_5d_total)
 .join(average_tutor_rating_over_5d_cl)
 .join(average_tutor_rating_over_5d_wf)[['student_id', 
                                         'started_at', 
                                         'average_student_rating_5d_total',
                                         'average_student_rating_over_5d_cl',
                                         'average_student_rating_over_5d_wf']]
 .to_csv(config_dir.joinpath(f'student_ratings.csv'))
)


# ## Student Start Date

# The earliest date for a each `student_id`.

# In[ ]:


student_start_date = feature_extraction.get_student_start_date(df_merged)


# In[ ]:


utils.save_object(student_start_date, 'student_start_date', column_dir)


# ## Tutor Age In Years

# Create a `pd.DataFrame` first to conveniently drop the blanks.

# In[ ]:


tutor_dates_df = (df_merged[['started_at', 'tutor_birth_year']]
                  .replace(0, np.nan)
                  .dropna(how='any', axis='rows')
                 )


# In[ ]:


tutor_age = tutor_dates_df.started_at.dt.year - tutor_dates_df.tutor_birth_year


# In[ ]:


tutor_age.head()


# In[ ]:


utils.save_object(tutor_age, 'tutor_age', column_dir)


# ## Tutor Experience in Days

# In[ ]:


tutor_days_df = (df_merged[['started_at', 'tutor_start_date']]
                         .replace(0, np.nan)
                         .dropna(how='any', axis='rows')
                        )


# In[ ]:


tutor_experience_days = (tutor_days_df.started_at - tutor_days_df.tutor_start_date).dt.days


# In[ ]:


tutor_experience_days.head()


# In[ ]:


utils.save_object(tutor_experience_days, 'tutor_experience_days', column_dir)


# ## Tutor Experience by Number of Sessions

# ### Split by Service

# Rather than using the elapsed time since the tutor joined, we could also use the number of sessions so far.

# In[ ]:


tutor_sessions = feature_extraction.expanding_count(
    df = df_merged,
    grouping_var = 'tutor_id',
    split_by_service = True,
)


# In[ ]:


for key, value in tutor_sessions.items():
    utils.save_object(value, key, column_dir)


# In[ ]:


tutor_sessions['tutor_num_sessions_cl'].head()


# ### Combined CL and WF

# In[ ]:


tutor_sessions_total = feature_extraction.expanding_count(
    df = df_merged,
    grouping_var = 'tutor_id',
    split_by_service = False,
)


# In[ ]:


utils.save_object(tutor_sessions_total, 'tutor_sessions_total', column_dir)


# In[ ]:


tutor_sessions_total.head()


# ## Student Experience by Number of Sessions

# ### Split by Service

# Rather than using the elapsed time since the tutor joined, we could also use the number of sessions so far.

# In[ ]:


student_sessions = feature_extraction.expanding_count(
    df = df_merged,
    grouping_var = 'student_id',
    split_by_service = True,
)


# In[ ]:


for key, value in student_sessions.items():
    utils.save_object(value, key, column_dir)


# In[ ]:


student_sessions['student_num_sessions_cl'].head()


# ### Combined CL and WF

# In[ ]:


student_sessions_total = feature_extraction.expanding_count(
    df = df_merged,
    grouping_var = 'student_id',
    split_by_service = False,
)


# In[ ]:


utils.save_object(student_sessions_total, 'student_sessions_total', column_dir)


# ## Guess the Sex By First Name

# In[ ]:


sex_guess = feature_extraction.guess_sex(df_merged.first_name)
sex_guess.name = 'sex_guess'


# In[ ]:


sex_guess.head()


# In[ ]:


utils.save_object(sex_guess, 'sex_guess', column_dir)


# # Remove Duplicate Comments

# Because there is a cost to calling the api (monetary and temporal), it would be economical to remove columns that have duplicate comments. The duplicate analysis is done on the lemmatised comments and with stopwords removed. These two steps ensures that the following two examples are treated as duplicates:
# - "I was waiting for 1 hour."
# - "I waited for 1 hour."
# 
# The reason is that these two sentences are merely different representations of the same intent. Additionally, the sentiment scores would be the same.

# In[ ]:


test_text1 = "I was waiting for 1 hour."
test_text2 =  "I waited for 1 hour."


# In[ ]:


print(feature_extraction.calc_sentiment(pd.Series(test_text1), 'textblob'))
print(feature_extraction.calc_sentiment(pd.Series(test_text2), 'textblob'))


# In[ ]:


print(feature_extraction.calc_sentiment(pd.Series(test_text1), 'vader'))
print(feature_extraction.calc_sentiment(pd.Series(test_text2), 'vader'))


# The above analysis shows that the both `TextBlob` and `VADER` score the two sentences neutrally and equally.

# In[ ]:


orth_list = []
lemma_list = []
rm_sw = []
for token in nlp(test_text1):
    orth_list.append(token.orth_)
    lemma_list.append(token.lemma_)
    if not token.is_stop:
        rm_sw.append(token.lemma_)

print(pd.DataFrame({'token': orth_list, 'lemma': lemma_list}))
print('Removed Stopwords: ', rm_sw)


# In[ ]:


orth_list = []
lemma_list = []
rm_sw = []
for token in nlp(test_text2):
    orth_list.append(token.orth_)
    lemma_list.append(token.lemma_)
    if not token.is_stop:
        rm_sw.append(token.lemma_)

print(pd.DataFrame({'token': orth_list, 'lemma': lemma_list}))
print('Removed Stopwords: ', rm_sw)


# The above shows that the two sentences become `['wait', '1', 'hour', '.']`. Of course the text that gets passed into the API would be the natural language.

# In[ ]:


student_comment_lemmatised_nostopwords_nopunct = student_comment_processed.apply(lambda x: ' '.join([token.lemma_ for token in x if not token.is_punct and not token.is_stop]))


# In[ ]:


utils.save_object(
    student_comment_lemmatised_nostopwords_nopunct,
    'student_comment_lemmatised_nostopwords_nopunct',
    column_dir)


# In[ ]:


data_df_comments['student_comment_lemmatised_nostopwords_nopunct'] = data_df_comments.student_comment_processed.apply(lambda x: ' '.join([token.lemma_ for token in x if not token.is_punct and not token.is_stop]))


# In[ ]:


student_comment_processed.iloc[0]


# In[ ]:


student_comment_lemmatised_nostopwords_nopunct.iloc[0]


# In[ ]:


len(student_comment_lemmatised_nostopwords_nopunct)


# In[ ]:


student_comment_deduplicated = process_luis_response.drop_semantic_duplicates(student_comment_processed)
len(student_comment_deduplicated)


# In[ ]:


utils.save_object(
    student_comment_deduplicated,
    'student_comment_deduplicated',
    column_dir
    )


# This is to create a series of `True` for duplicated values and `False` otherwise. The parameter `keep='first'` ensures that the first of the duplicated comment has a value of 'false', because we want to keep at least one of these.

# In[ ]:


mask_duplicated_lemmas = data_df_comments.duplicated(subset = 'student_comment_lemmatised_nostopwords_nopunct', keep = 'first')
mask_duplicated_lemmas.sum()


# There are 18523 rows that are duplicated. Let's see what these look like.

# The unprocessed version looks like this.

# In[ ]:


data_df_comments.student_comment[mask_duplicated_lemmas][:10]


# The lemmatised version:

# In[ ]:


data_df_comments.student_comment_lemmatised_nostopwords_nopunct[mask_duplicated_lemmas][:10]


# After removing the duplicates, see if the index of the DataFrame is unique.

# In[ ]:


len_unique_comments = len(data_df_comments.student_comment_apostrophe[~mask_duplicated_lemmas])
len_unique_comments


# In[ ]:


data_df_comments.student_comment_lemmatised_nostopwords_nopunct.to_csv(Path.cwd().joinpath('OUTPUT').joinpath('student_comment_lemmatised_nostopwords_nopunct.csv'), header=True)


# # Gender Guesser

# This section uses the [`gender_guesser`](https://pypi.org/project/gender-guesser) package to guess the gender of the students based on their first names.

# In[ ]:


sex_guess = df_merged.first_name.apply(feature_extraction.guess_sex)


# In[ ]:


print(df_merged.first_name.head())


# In[ ]:


sex_guess.head()


# In[ ]:


utils.save_object(sex_guess, 'sex_guess', column_dir)


# In[ ]:


sex_guess.value_counts()


# In[ ]:


title = 'Count of Sex'
x_label = 'Sex'
y_label = 'Count'
ax = sns.countplot(sex_guess)
ax.set(title=title
       ,xlabel=x_label
       ,ylabel=y_label)


# ## `unknown` Sex

# In[ ]:


df_merged[sex_guess == 'unknown']['first_name'].head(20)


# It seems like the `unknown` are predominantly female.

# # Merge Columns to the DataFrame

# In[ ]:


df_features = df_merged.copy(deep=True)


# In[ ]:


for filename in column_dir.glob('*'):
    print(f'Joining {filename.stem}')
    df_features = utils.add_column(
        df_features,
        column_dir,
        str(filename.stem))


# ## Joining Comments Data with the Non-Comments Data

# The two data frames (`data_df` and `data_df_comments`) have overlapping columns. A useful join only adds the extra columns.

# In[ ]:


print(data_df_comments.shape)
print(data_df.shape)


# Find the columns in `data_df_comments` that are not in `data_df`.

# In[ ]:


join_columns = set(data_df_comments) - set(data_df)
print(join_columns)


# In[ ]:


df = data_df.join(data_df_comments[join_columns])


# In[ ]:


df.shape


# In[ ]:


# Save pickle
with open(Path.cwd().joinpath('OUTPUT').joinpath('df_full'), 'ab') as f:
    pickle.dump(df, f)


# In[ ]:


with open (Path.cwd().joinpath('OUTPUT').joinpath('df_full'), 'rb') as f:
    df = pickle.load(f)

