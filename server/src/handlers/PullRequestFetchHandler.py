# PullRequestFetchHandler.py

import os
from flask import Flask, request
from parse_pull_request_event import parse_pull_request_event, MissingFieldError
from pymongo import MongoClient
import logging

app = Flask(__name__)

# Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['pull_requests']
collection = db['events']

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/github/webhook', methods=['POST'])
def webhook_handler():
    payload = request.get_json()
    if payload.get('action') == 'opened':
        try:
            event = parse_pull_request_event(payload)
            process_event(event)
        except MissingFieldError as e:
            logger.error(f'Missing field error: {e}')
            return f'Error: {e}', 400
        except Exception as e:
            logger.exception(f'An unexpected error occurred: {e}')
            return 'Internal Server Error', 500
    return '', 200

def process_event(event):
    try:
        collection.insert_one(event.__dict__)
        logger.info(f'Pull request event saved: {event.pull_request_id}')
    except Exception as e:
        logger.exception(f'Failed to save event: {e}')
        raise

# Ensure client is closed when the application exits
@app.teardown_appcontext
def close_connection(exception):
    client.close()