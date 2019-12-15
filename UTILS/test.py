import unittest
from pathlib import Path
import utils
import feature_extraction
import xlwings as xw
import pandas as pd
import datetime

# %%
import importlib
importlib.reload(utils)
importlib.reload(feature_extraction)
# %%
class TestData(unittest.TestCase):
    @unittest.skip("Test passed already")
    def test_that_rename_columns_produces_correct_columns(self):
        cl_columns = ['session_id','started_at','duration_seconds','wait_seconds','tutor_id','tutor_country','student_id','subject_student','skillset','year_level','study_type','esl','first_name','student_comment','student_rating','client_id','client_country','client_state','client_region','client_type_desc','app_version','platform_type','platform_version','device_type','qualifications','tutor_birth_year','tutor_start_date']
        wf_columns = ['session_id','started_at','pay_minutes','completed_at','tutor_id','tutor_country','student_id','year_level','esl','first_name','study_type','student_comment','student_rating','client_id','client_country','client_state','client_region','client_type_desc','sub_document_type','first_viewed_by_student_on_mobile','qualifications','tutor_birth_year','tutor_start_date']        
        
        data_dir = Path.cwd().parents[0].joinpath('DATA')
        config_dir = Path.cwd().parents[0].joinpath('CONFIG')
        
        expected_columns_dict = {'cl': cl_columns, 'wf': wf_columns}
        
        wb = xw.Book(str(data_dir.joinpath('CL_20190823.xlsx')))
        cl_df = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
    
        wb = xw.Book(str(data_dir.joinpath('WF_20190826.xlsx')))
        wf_df = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
        
        #wb.app.quit()
        
        columns_df = pd.read_csv(config_dir.joinpath('columns.csv'))
        
        for service, df in {'cl': cl_df, 'wf': wf_df}.items():
            
            renamed_df = utils.rename_columns(df, service, columns_df)
            self.assertListEqual(list(renamed_df.columns), expected_columns_dict[service])
    
    
    @unittest.skip("Test passed already")
    def test_the_shapes_of_dataframe_after_load_data(self):
        data_dir = Path.cwd().parents[0].joinpath('DATA')
        
        cl_df, wf_df = utils.load_data(data_dir)
        
        cl_shape = (107798, 27)
        wf_shape = (398489, 23)
        
        self.assertTupleEqual(cl_df.shape, cl_shape)
        self.assertTupleEqual(wf_df.shape, wf_shape)
        
    
    @unittest.skip("Test passed already")
    def test_that_configfile_is_two_dimensional(self):
        config_dir = Path.cwd().parents[0].joinpath('CONFIG')
        for filename in ['columns', 'columns.csv', 'mapping_intents.csv']:
            config_df = utils.load_config(filename, config_dir)
            self.assertEqual(len(config_df.shape), 2)
        
    
    @unittest.skip("Test passed already")
    def test_that_data_has_been_correctly_merged(self):
        pass
    
    
    @unittest.skip("Test passed already")
    def test_excel_float_to_date_converter(self):
        float_dates = [42814.214912, 43224.481656, 43590.461014]
        benchmark_dates = ['20170320 050928',
                           '20180504 113335',
                           '20190505 110351']
        
        for i, date in enumerate(float_dates):
            converted_date = utils.convert_excel_float_to_date(date).strftime('%Y%m%d %H%M%S')
            self.assertEqual(converted_date, benchmark_dates[i])


    def test_that_columns_dtypes_have_been_changed_correctly(self):
        #TODO
        # Not complete
        cl_types = ['int64','datetime64','int64','int64','int64','category','int64','category','category','category','category','category','str','str','ordinal','int64','category','category','category','category','str','category','category','category','list','int64','datetime64']
        wf_types = ['int64','datetime64','int64','datetime64','int64','category','int64','ordinal','bool','str','category','str','ordinal','int64','category','category','category','category','category','bool','list','int64','datetime64'] 
        
        data_dir = Path.cwd().parents[0].joinpath('DATA')
        config_dir = Path.cwd().parents[0].joinpath('CONFIG')
        
        expected_columns_dict = {'cl': cl_types, 'wf': wf_types}
        
        wb = xw.Book(str(data_dir.joinpath('CL_20190823.xlsx')))
        cl_df = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
    
        wb = xw.Book(str(data_dir.joinpath('WF_20190826.xlsx')))
        wf_df = wb.sheets[0].used_range.options(pd.DataFrame, index = False, header = True).value
        
        data_dict = {'cl': cl_df, 'wf': wf_df}
        
        column_type_df = pd.read_csv(config_dir.joinpath('column_types.csv'))
        
        for service, df in data_dict.items():
            df = df.apply(lambda x: utils.set_column_type(x, service, column_type_df))
            


class TestFeatureExtraction(unittest.TestCase):
    def test_get_comment_length(self):
        comment1 = 'took 1 hour to finally get connected'
        comment2 = ''
              
        comment1_len_word = feature_extraction.get_comment_length(comment1, 'word')
        comment1_len_character = feature_extraction.get_comment_length(comment1, 'character')
        comment2_len_word = feature_extraction.get_comment_length(comment2, 'word')
        comment2_len_character = feature_extraction.get_comment_length(comment2, 'character')    
        
        self.assertEqual(comment1_len_word, 7)
        self.assertEqual(comment1_len_character, 36)
        self.assertEqual(comment2_len_word, 0)
        self.assertEqual(comment2_len_character, 0)


    def test_rolling_mean(self):
        data_dir = Path.cwd().parents[0].joinpath('DATA')
                
        wb = xw.Book(str(data_dir.joinpath('CL_20190823.xlsx')))
        cl_df = (wb
                 .sheets[0]
                 .used_range
                 .options(pd.DataFrame, 
                          index = False, 
                          header = True)
                 .value
                 )
                 
        
        # A couple of tutor ids who are known to have had multiple sessions and
        # whose student rating is not blank
        tutor_ids = [1358, 1463]
        df = cl_df.query('tutor_id in @tutor_ids and student_rating > 0')
        
        # Rename column to match with the default in the rolling_mean function
        df.rename(
                columns={'started_at_utc': 'started_at'}, 
                inplace=True
                )
        
        # Format the started_at column as datetime
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
        
        df['started_at'] = df.started_at.apply(convert_excel_float_to_date)
        
        col = 'student_rating'
        group = 'tutor_id'
        window = '2d'
        
        average_rating = feature_extraction.rolling_mean(
                df = df,
                col = col,
                group = group,
                window = window,
                                        )

    
# =============================================================================
# if __name__ == "__main__":
#     unittest.main()
# =============================================================================
