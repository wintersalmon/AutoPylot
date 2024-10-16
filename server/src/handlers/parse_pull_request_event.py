from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

class MissingFieldError(Exception):
    """필수 필드가 누락되었을 때 발생하는 예외."""
    def __init__(self, field_name: str):
        super().__init__(f"필수 필드가 누락되었습니다: {field_name}")

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

def extract_pull_request_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        pull_request = payload['pull_request']
        repository = payload['repository']
        data = {
            'repository_name': repository['full_name'],
            'pull_request_number': pull_request['number'],
            'pull_request_id': pull_request['id'],
            'head_sha': pull_request['head']['sha'],
            'author': pull_request['user']['login'],
            'title': pull_request['title'],
            'body': pull_request.get('body', ''),
            'created_at': pull_request['created_at'],
            'html_url': pull_request['html_url']
        }
        return data
    except KeyError as e:
        raise MissingFieldError(str(e))

def create_pull_request_event(data: Dict[str, Any]) -> PullRequestEvent:
    try:
        event = PullRequestEvent(
            repository_name=data['repository_name'],
            pull_request_number=data['pull_request_number'],
            pull_request_id=data['pull_request_id'],
            head_sha=data['head_sha'],
            author=data['author'],
            title=data['title'],
            body=data['body'],
            created_at=datetime.strptime(data['created_at'], "%Y-%m-%dT%H:%M:%SZ"),
            html_url=data['html_url']
        )
        return event
    except KeyError as e:
        raise MissingFieldError(str(e))

def parse_pull_request_event(payload: Dict[str, Any]) -> PullRequestEvent:
    data = extract_pull_request_data(payload)
    event = create_pull_request_event(data)
    return event
