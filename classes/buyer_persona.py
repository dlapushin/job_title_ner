import pickle5 as pickle
import string
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter, OrderedDict
from tqdm import tqdm
tqdm.pandas()

with open('title_classifier_serial', 'rb') as f:
    classifier = pickle.load(f)

with open('vectorizer_serial', 'rb') as f:
    vectorizer = pickle.load(f)

class Buyer_Persona:
    
    def __init__(self):
        self.df = None

    def data_ingest(self, file_path, job_title_field):
        # Read the data from the given file_path into a pandas DataFrame
        # Assumes the file is either .csv or .xlsx
        if file_path.endswith(".csv"):
            print('Reading data file: ' + file_path)
            self.df = pd.read_csv(file_path, low_memory=False)
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        elif file_path.endswith(".xlsx"):
            self.df = pd.read_excel(file_path, low_memory=False)
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

        # Validate if the job_title_field exists in the DataFrame
        if job_title_field not in self.df.columns:
            raise ValueError(f"Job title field '{job_title_field}' not found in the DataFrame.")

    def title_adjuster(self, job_title):
        translating = str.maketrans('', '', string.punctuation)
        title_adj = str.lower(str(job_title))
        title_adj = title_adj.translate(str.maketrans('', '', string.punctuation))

        return title_adj
    
    def title_clean(self, job_title_field):
        if self.df is None:
            raise ValueError("DataFrame is empty. Please use data_ingest to load data first.")
        print('Cleaning data in field: ' + job_title_field)
        
        clean_title_field = job_title_field + '_clean' ## creates new field based on the title field and adds "_clean"
        self.df[clean_title_field] = self.df[job_title_field].progress_apply(lambda x: self.title_adjuster(x))
        self.clean_title = clean_title_field
        
    def predict_category(self, token):
        token_features = vectorizer.transform([token])
        prediction = classifier.predict(token_features)

        return prediction[0]        
    
    ## Create a master dictionary of all the tokens found in titles
    ## Predict the POS (RES or FUN) for each token 
    ## Persist the resulting token:pos dictionary
    def master_token_to_odict(self):  
        self.series_title = self.df[self.clean_title].tolist()

        print('Flattening tokens into list format:')
        self.master_token = []
        for t in tqdm(self.series_title):
            token_split_list = t.split(' ')

            for t_elem in token_split_list:
                self.master_token.append(t_elem.strip())
                
        self.odict_sorted = OrderedDict(Counter(self.master_token).most_common())

        print('Creating dictionary of token to POS:')
        self.odict_pos = {}
        for key in tqdm(self.odict_sorted):
            self.odict_pos[key] = {'count': self.odict_sorted[key],
                            'pos': self.predict_category(key)
                           }

    def jt_tokenized_dict(job_title_text):

        df_job_title_breakdown = pd.DataFrame()
        dict_pos = {'RES': [],
                    'FUN': []
                   }

        for token in job_title_text.split(' '):
            predicted_pos = predict_category(token.lower().str.strip())

            if(predicted_pos == 'RES'):
                dict_pos['RES'].append(token)
            if(predicted_pos == 'FUN'):
                dict_pos['FUN'].append(token)

        dict_pos['RES'] = ' '.join(sorted(dict_pos['RES']))
        dict_pos['FUN'] = ' '.join(sorted(dict_pos['FUN']))
        
        return dict_pos

    def tokenized_dict_to_col(df):

        dict_res_fun = jt_tokenized_dict(df['Title_adj'])
        df['RES'] = dict_res_fun['RES']
        df['FUN'] = dict_res_fun['FUN']
        return df

    def jt_tokenized_lists_v2(self, input_title_clean):
        RES = []
        FUN = []

        for token in input_title_clean.split(' '):

            try:
                predicted_pos = self.odict_pos[token]['pos']
            except:
                predicted_pos = None

            if(predicted_pos == 'RES'):
                RES.append(token)
            if(predicted_pos == 'FUN'):
                FUN.append(token)
        
        return [RES, FUN]        

    def pos_mapper(self):
        # Apply the lambda function to the DataFrame
        print('Parsing field ' + self.clean_title + ' into RES and FUN components.  May take a few moments.')
        self.df = pd.concat([self.df, self.df.progress_apply(lambda x: self.jt_tokenized_lists_v2(x[self.clean_title]), axis=1, result_type = 'expand')], axis=1)
        self.df.rename(columns={0:'RES', 1:'FUN'}, inplace=True)
        self.df['RES_flat'] = self.df['RES'].progress_apply(lambda x: ' '.join(x))
        self.df['FUN_flat'] = self.df['FUN'].progress_apply(lambda x: ' '.join(x))
        
    def sorted_pos_token(self, pos):
        token_counts = pd.DataFrame(self.df.groupby(pos).size().sort_values(ascending=False))
        token_counts.reset_index(inplace=True)
        token_counts.columns = [pos, 'count']

        return token_counts

    def pos_sorter(self):
        print('Sorting RES and FUN values')
        self.list_res = list(self.sorted_pos_token('RES_flat')['RES_flat'])
        self.list_res.remove('')
                            
        self.list_fun = list(self.sorted_pos_token('FUN_flat')['FUN_flat'])
        self.list_fun.remove('')
     
    def top_pos_sorter(self, n):
        print('Finding top RES and FUN values')
 
        self.top_res = self.list_res[0:n]
        self.top_fun = self.list_fun[0:n]
        
    def top_pos_match(self):

        filter_res = (self.df['RES_flat'].isin(self.top_res))
        filter_fun = (self.df['FUN_flat'].isin(self.top_fun))

        self.df_top_pos_matches = self.df[filter_res & filter_fun]
        self.df_top_pos_matches.reset_index(inplace=True, drop=True)
        
    def top_pos_match_pchart(self):
        
        self.df_list = self.df_top_pos_matches[['RES_flat','FUN_flat']]
        self.df_list.reset_index(inplace=True, drop=True)
        data = self.df_list.values.tolist()

        df = pd.DataFrame(data, columns = ['RES', 'FUN'])
        fig = px.parallel_categories(df)
        fig.update_layout(width = 1000,
                          height = 1000, 
                          title = 'Parallel categories for Top Responsibility Levels (RES) and Functions (FUN)')
        fig.show()
        
    def top_pos_match_heatmap(self):
        
        df_focus = self.df_top_pos_matches.groupby(['RES_flat','FUN_flat']).count().reset_index()[['RES_flat','FUN_flat','Title']]
        df_focus = df_focus.pivot(index='RES_flat', columns='FUN_flat')['Title'].fillna(0)

        fig = px.imshow(df_focus, x=new_df.columns, y=new_df.index, color_continuous_scale=px.colors.sequential.GnBu)
        fig.update_layout(width = 1000,
                          height = 1000,
                         title = 'Heatmap for top Responsibility Levels (RES) and Functions (FUN)')
        fig.update_xaxes(title_text='Top Functions (FUN)')
        fig.update_yaxes(title_text='Top Responsibility Levels (RES)')
        fig.show()
        

# Example usage:
buyer_persona = Buyer_Persona()
buyer_persona.data_ingest('my_crm_contacts.csv', 'Title')
buyer_persona.title_clean('Title')
buyer_persona.master_token_to_odict()

buyer_persona.pos_mapper()
buyer_persona.pos_sorter()
buyer_persona.top_pos_sorter(20)
buyer_persona.top_pos_match()

display(buyer_persona.top_pos_match_heatmap())
display(buyer_persona.top_pos_match_pchart())
