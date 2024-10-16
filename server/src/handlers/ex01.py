import json
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PullRequestEvent:
    repository_name: str
    pull_request_number: int
    pull_request_id: int
    head_sha: str
    author: str
    title: str
    body: str
    created_at: datetime
    html_url: str

def parse_pull_request_event(payload: Dict[str, Any]) -> PullRequestEvent:
    pull_request = payload['pull_request']
    repository = payload['repository']

    event = PullRequestEvent(
        repository_name=repository['full_name'],
        pull_request_number=pull_request['number'],
        pull_request_id=pull_request['id'],
        head_sha=pull_request['head']['sha'],
        author=pull_request['user']['login'],
        title=pull_request['title'],
        body=pull_request['body'],
        created_at=datetime.strptime(pull_request['created_at'], "%Y-%m-%dT%H:%M:%SZ"),
        html_url=pull_request['html_url']
    )
    return event

# Flask를 사용하여 Webhook 엔드포인트 구현 예시
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    payload = request.get_json()
    if payload['action'] == 'opened':
        event = parse_pull_request_event(payload)
        process_event_async(event)
    return '', 200

def process_event_async(event: PullRequestEvent) -> None:
    # 비동기 처리를 위한 로직 구현 (예: 메시지 큐에 이벤트 전송)
    pass