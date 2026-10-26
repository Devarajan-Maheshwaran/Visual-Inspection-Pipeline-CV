FROM python:3.12.15-slim

WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
COPY . /code
RUN python -m src.train && python -m src.export

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]