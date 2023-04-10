import pickle
import logging
import pandas as pd
import openai
import os

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')

class RestaurantModel:
    
    def __init__(self, conn):
        self.cache = pickle.load(open('cache.pkl', 'rb'))
        self.conn = conn
        self._generate_database_map(
            categorical_columns=['MENU_ITEMS.CATEGORY', 'EMPLOYEE_STATUS.POSITION'], 
            version='categories'
        )
        # self._generate_database_map(categorical_columns=[], version='datatypes')
        logging.basicConfig(
            filename='restaurant_analyst.log', 
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s',
        )

    def _generate_database_map(self, categorical_columns, version):
        '''
        Generates a database map for the given SQLite connection.

        This method generates a self.database_map attribute for the SQLite connection. 
        It reads all tables from the database, retrieves the column information for each 
        table and formats it as "# <TABLE_NAME>(<COLUMN_NAME>:<COLUMN_TYPE>)".
        
        This information is then concatenated and saved as self.database_map attribute for future use.
        
        We are still testing the best prompt to give ChatGPT – so it accepts different "version" inputs:
        - datatypes: each column will have its datatype next to it
        - categories: each categorical column will have a list of possible values
        '''
        assert version in ['categories', 'datatypes'], 'Unsupported database map generator version!'
        all_tables = pd.read_sql('select * from sqlite_schema', self.conn)
        schema_str = ''
        for table in all_tables.tbl_name.to_list():
            cols = pd.read_sql(f'PRAGMA table_info({table})', self.conn)
            if version == 'datatypes':
                cols_dict = cols[['name', 'type']].to_dict('records')
                col_and_type = ', '.join([f"{x['name']}:{x['type']}" for x in cols_dict])
                schema_str += f"# {table.upper()}({col_and_type})\n"
            elif version == 'categories':
                cols_and_cats_dict = {}
                for c in cols.name:
                    if f'{table.upper()}.{c.upper()}' in categorical_columns:
                        cols_and_cats_dict[c] = pd.read_sql(f'SELECT DISTINCT {c} FROM {table}', self.conn)[c].to_list()
                    else:
                        cols_and_cats_dict[c] = []
                col_and_cats_str = ', '.join([col if not cats else f'{col}:{cats}' for col, cats in cols_and_cats_dict.items()])
                schema_str += f"# {table.upper()}({col_and_cats_str})\n"
        self.database_map = schema_str

    @property
    def query_prompt(self):
        prompt = f'### SQLite tables, with their properties:\n#\n{self.database_map}#\n### A query to answer "{self.question}"\nSELECT'
        return prompt

    def obtain_query(self):
        '''
        Gets a SQL query from the openAI endpoint using the self.query_prompt attribute.
        
        Note: we prompt the model with the word SELECT. We therefore need to prepend 
        it to the result.
        '''
        logging.info(f'Running obtain_query, attempt number {self.attempt_number}')
        if 'query' in self.cache.get(self.question, {}) and self.attempt_number == 1:
            query = self.cache[self.question]['query']
            logging.info('Using cached query prompt')
        else:
            response = openai.Completion.create(
              model="text-davinci-003",
              prompt=self.query_prompt,
              temperature=0,
              max_tokens=200,
              top_p=1.0,
              frequency_penalty=0.0,
              presence_penalty=0.0,
              stop=["#", ";"]
            )
            query = 'SELECT' + response['choices'][0]['text'] 
            query = ' '.join(query.split())
            logging.info('Obtained query prompt')
        return query

    def load_data(self, query):
        '''
        Executes the query obtained from openAI against the database.
        Saves the result in a DataFrame and returns it.
        '''
        logging.info(f'Running load_data, attempt number {self.attempt_number}')
        if 'df' in self.cache.get(self.question, {}) and self.attempt_number == 1:
            df = self.cache[self.question]['df']
            logging.info('Using cached data')
        else:
            df = pd.read_sql(query, self.conn)
            logging.info('Loaded data')
        return df

    def generate_interpretation_prompt(self, df):
        question_str = f'### Original question: {self.question}'
        data_str = '#' + '\n#'.join([str(x) for x in df.to_dict('records')])
        prompt_str = '### Write a response the the question:\nThe'
        full_prompt = f'{question_str}\n### Data which fully answers the question:\n{data_str}\n{prompt_str}'
        return full_prompt

    def obtain_interpretation(self, df):
        '''
        Gets the user response from the openAI endpoint by giving it the DataFrame of 
        results and the user's original question.
        
        Note: we prompt the model with the word The. We therefore need to prepend 
        it to the result.
        '''
        logging.info(f'Running obtain_interpretation')
        if 'interpretation' in self.cache.get(self.question, {}):
            interpretation = self.cache[self.question]['interpretation']
            logging.info('Using cached interpretation')
        else:
            interpretation_prompt = self.generate_interpretation_prompt(df)
            response = openai.Completion.create(
              model="text-davinci-003",
              prompt=interpretation_prompt,
              temperature=0,
              max_tokens=200,
              top_p=1.0,
              frequency_penalty=0.0,
              presence_penalty=0.0,
              stop=["#", ";"] # do we need this here?
            )
            interpretation = 'The' + response['choices'][0]['text'] 
            self.cache[self.question]['interpretation'] = interpretation
            logging.info('Obtained interpretation')
        return interpretation
    
    def ask(self, question):
        '''
        Method to obtain the interpretation of a given question.

        Args:
        - question (str): The question for which the interpretation needs to be obtained.

        Returns:
        - interpretation (str): The interpretation of the question obtained from the data.

        This method first checks if the given question is present in the cache and if an interpretation is already
        present for the question. If so, it returns the cached interpretation. If not, it obtains the query for the
        question, loads the necessary data based on the query, and generates an interpretation prompt for the data.
        The obtained interpretation is then cached for future use and returned.
        - 
        '''
        
        self.attempt_number = 1 # this tracks how many tries we gave ChatGPT to generate the correct query.
        query_compiles = False # have we obtained an executable query from ChatGPT?
        query_returns_data = False # does the query return any data?
        self.question = question
        
        logging.info(f'New question: {question}')
        
        if question not in self.cache:
            self.cache[question] = {}
        
        while self.attempt_number <= 3 and not query_returns_data and not query_compiles:
            # step 1: obtain query
            query = self.obtain_query()
            
            # step 2: load DF
            try:
                df = self.load_data(query)
            except Exception as e:
                logging.warning(f'Error in running query!')
                self.attempt_number += 1 
                continue
            else:
                query_compiles = True
                
            query_returns_data = len(df) > 0
            self.attempt_number += 1 
            
        if not query_compiles:
            print('Unable to obtain an executable query...')
        else:
            # cache obtained results:
            self.cache[question]['timestamp'] = datetime.now()
            self.cache[question]['query'] = query
            self.cache[question]['df'] = df

            # step 3: return interpretation
            interpretation = self.obtain_interpretation(df)

            # step 4: persist the cache
            pickle.dump(self.cache, open('cache.pkl', 'wb'))

            print(interpretation)