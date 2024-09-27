# docker/Dockerfile
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
RUN pip install poetry

# 프로젝트 파일 복사
COPY pyproject.toml poetry.lock /app/

# 의존성 설치
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# 애플리케이션 코드 복사
COPY src /app/src

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "src/main.py"]