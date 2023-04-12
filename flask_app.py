from flask import Flask, request
from model import RestaurantModel
import sqlite3

from sqlalchemy import create_engine

DB_HOST = 'database-1.cmqi9a22jijy.eu-west-2.rds.amazonaws.com'

DB_NAME = 'postgres'
DB_PORT = 5432
DB_USER = 'postgres'
DB_PASS = 'master-password'

DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URI)

app = Flask(__name__)
restaurant_model = RestaurantModel(
    # conn=sqlite3.connect('restaurant.db', check_same_thread=False),
    conn=engine,
    secret_protected=True,
    db_type='postgresql'
)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    secret = data.get('secret')
    answer = restaurant_model.ask(question=question, secret=secret)
    return {'answer': answer}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    