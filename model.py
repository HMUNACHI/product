import pickle
import logging
import pandas as pd
import openai
import os
import re
import sys

from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')

def get_all_tables(conn, db_type):
    """
    gets all tables from the database and returns them as a list
    """
    if db_type == 'sqlite':
        all_tables = pd.read_sql('select tbl_name from sqlite_schema', conn)
    elif db_type == 'postgresql':
        all_tables = pd.read_sql("SELECT table_name FROM information_schema.tables where table_schema = 'public'", conn)
    else:
        raise ValueError('Unsupported database type!')
    return all_tables.iloc[:, 0].to_list()

def get_all_columns(conn, db_type, table_name):
    """
    gets all columns for a given table returns them as a DataFrame
    """
    if db_type == 'sqlite':
        all_columns = pd.read_sql(f'PRAGMA table_info({table_name})', conn)
    elif db_type == 'postgresql':
        all_columns = pd.read_sql(f"SELECT column_name as name, data_type as type FROM information_schema.columns WHERE table_name = '{table_name}'", conn)
    else:
        raise ValueError('Unsupported database type!')
    return all_columns

def parse_postgres_error(text):
    regex_pattern = r'\) (.*)\n'
    matches = re.search(regex_pattern, text)
    if matches:
        return matches.group(1)
    else:
        return None

def run_default_openai_completion(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      temperature=0,
      max_tokens=200,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=["#", ";"]
    )
    return response


class RestaurantModel:
    
    def __init__(self, conn, db_type, secret_protected=False):
        self.secret_protected = secret_protected
        self.db_type = db_type
        self.cache = pickle.load(open('cache.pkl', 'rb'))
        self.conn = conn
        self._generate_database_map(
            categorical_columns=['MENU_ITEMS.CATEGORY', 'EMPLOYEE_STATUS.POSITION'], 
        )
        logging.basicConfig(
            filename='restaurant_analyst.log', 
            level=logging.INFO,
            format='%(asctime)s %(levelname)-8s %(message)s',
        )

    def _generate_database_map(self, categorical_columns=[]):
        '''
        Generates a database map for the given SQL connection.

        This method generates a self.database_map attribute for the SQL connection. 
        It reads all tables from the database, retrieves the column information for each 
        table and formats it as "# <TABLE_NAME>(<COLUMN_NAME>:<COLUMN_TYPE>)".
        
        This information is then concatenated and saved as self.database_map attribute for future use.
        
        We are still testing the best prompt to give ChatGPT so it accepts different "version" inputs:
        - datatypes: each column will have its datatype next to it
        - categories: each categorical column will have a list of possible values
        '''
        all_tables = get_all_tables(self.conn, self.db_type)
        schema_str = ''
        for table in all_tables:
            cols = get_all_columns(self.conn, self.db_type, table)
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
        prompt = f'### {self.db_type} tables, with their properties:\n#\n{self.database_map}#\n### A query to answer "{self.question}"\nSELECT'
        return prompt

    def obtain_query(self, sql_compilation_error=None, last_run_query=None):
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
            if not sql_compilation_error and not last_run_query:
                response = run_default_openai_completion(self.query_prompt)
            else:
                prompt = f'''### {self.db_type} query:\n# {last_run_query}\n### SQL error:\n# {sql_compilation_error}\n### Fix the error.\nSELECT'''
                print(prompt)
                response = run_default_openai_completion(prompt)
            query = 'SELECT' + response['choices'][0]['text'] 
            query = ' '.join(query.split())
            # logging.info('Obtained query!')
            logging.info(f'Obtained query: {query}')
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
            response = run_default_openai_completion(interpretation_prompt)
            interpretation = 'The' + response['choices'][0]['text'] 
            self.cache[self.question]['interpretation'] = interpretation
            logging.info('Obtained interpretation')
        return interpretation
    
    def ask(self, question, secret=None):
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
        if self.secret_protected and secret != os.getenv('MODEL_SECRET'):
            # raise ValueError('Incorrect secret provided!')
            return 'Incorrect secret provided!'
        self.attempt_number = 1 # this tracks how many tries we gave ChatGPT to generate the correct query.
        sql_compilation_error = None # this tracks the error message from the SQL compilation error (none at first)
        last_run_query = None # this tracks the last query that was run (none at first)
        query_compiles = False # have we obtained an executable query from ChatGPT?
        query_returns_data = False # does the query return any data?
        self.question = question
        
        logging.info(f'New question: {question}')
        
        if question not in self.cache:
            self.cache[question] = {}
        
        while self.attempt_number <= 3 and not query_returns_data and not query_compiles:
            # step 1: obtain query
            query = self.obtain_query(sql_compilation_error=sql_compilation_error, last_run_query=last_run_query)
            
            # step 2: load DF
            try:
                df = self.load_data(query)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                sql_compilation_error = parse_postgres_error(str(exc_value))
                last_run_query = query
                logging.warning(f'Error in running query! ({e}))')
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

            return interpretation