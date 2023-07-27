def installs():  # you may need to install certain libraries

    import pickle5 as pickle
    import string
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    import pandas as pd
    import os

    with open('title_classifier_serial', 'rb') as f: ## This pretrained model file should also be in the directory
        classifier = pickle.load(f)

    with open('vectorizer_serial', 'rb') as f:  ## This vectorizer file should also be in the directory
        vectorizer = pickle.load(f)

class Buyer_Persona:
    def __init__(self):
        self.df = None

    def data_ingest(self, file_path, job_title_field):
        # Read the data from the given file_path into a pandas DataFrame
        # Assuming the file is either .csv or .xlsx
        if file_path.endswith(".csv"):
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

    def pos_map(self, job_title):
        # Apply the existing NLP model to get RES and FUN values
        res, fun = nlp_pos_mapper(job_title)
        return res, fun

    def title_adjuster(self, job_title):
        title_adj = str.lower(str(job_title))
        title_adj = title_adj.translate(str.maketrans('', '', string.punctuation))
        return title_adj
    
    def title_clean(self, job_title_field):
        if self.df is None:
            raise ValueError("DataFrame is empty. Please use data_ingest to load data first.")
        
        translating = str.maketrans('', '', string.punctuation)
        self.df['Title_clean'] = self.df[job_title_field].apply(lambda x: self.title_adjuster(x))

    def title_clean(self, job_title_field):
        if self.df is None:
            raise ValueError("DataFrame is empty. Please use data_ingest to load data first.")
        
        translating = str.maketrans('', '', string.punctuation)
        self.df['Title_clean'] = self.df[job_title_field].apply(lambda x: self.title_adjuster(x))
        
    def predict_category(self, token):
        token_features = vectorizer.transform([token])
        prediction = classifier.predict(token_features)
        return prediction[0]        

    def jt_tokenized_lists(self, input_title_clean):
        
        RES = []
        FUN = []

        for token in input_title_clean.split(' '):
            predicted_pos = predict_category(token)  ## TOKEN MUST BE IN LOWER CASE!!!

            if(predicted_pos == 'RES'):
                RES.append(token)
            if(predicted_pos == 'FUN'):
                FUN.append(token)

        return [RES, FUN]

    def pos_mapper(self, job_title_field):
        # Apply the lambda function to the DataFrame
        self.df = pd.concat([self.df, self.df.apply(lambda x: self.jt_tokenized_lists(x[job_title_field]), axis=1, result_type = 'expand')], axis=1)


### Example usage:
# installs()
# buyer_persona = Buyer_Persona()
# buyer_persona.data_ingest('sfdc_contacts.csv', 'Title')

# buyer_persona.title_clean('Title')
# buyer_persona.pos_mapper('Title_clean')
# print(buyer_persona.df)
