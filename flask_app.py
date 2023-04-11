from flask import Flask, request
from model import RestaurantModel
import sqlite3

app = Flask(__name__)
restaurant_model = RestaurantModel(
    conn=sqlite3.connect('restaurant.db', check_same_thread=False),
    secret_protected=True,
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