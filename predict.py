# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:01:49 2019

@author: Roger Yu

This script is used to produce predictions of ratings given the data in either
the CL or WF formats. There should only be one file in the DATA/PREDICT/ 
folder.
"""

from pathlib import Path
import pandas as pd
from UTILS import utils
import pickle

# =============================================================================
# Set up folders
# =============================================================================

data_dir = Path.cwd().joinpath('DATA').joinpath('PREDICT')
output_dir = Path.cwd().joinpath('OUTPUT')
prediction_dir = output_dir.joinpath('PREDICTION')
model_dir = output_dir.joinpath('MODELS')

# =============================================================================
# Load data
# =============================================================================

for file in data_dir.iterdir():
    filename = file
    
    if 'cl' in filename.stem.lower():
        service == 'cl'
    elif 'wf' in filename.stem.lower():
        service == 'wf'
    else:
        raise Exception(f'The filename, {file}, does not have either "CL" or "WF".')
        
df = pd.read_excel(filename)

# =============================================================================
# Preprocess
# =============================================================================
df_preprocess = utils.preprocess(df, service)

X = df_preprocess.drop(labels='student_rating', axis='columns')

# =============================================================================
# Load model
# =============================================================================
model_path = model_dir.joinpath('xgb_regressor_randomizedsearchcv')
with open(model_path.as_posix(), 'rb') as file:
    model = pickle.load(file)

model.best_estimator_.predict(result_df)
