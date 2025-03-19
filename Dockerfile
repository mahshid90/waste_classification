FROM python:3.10.6-buster
WORKDIR /prod

COPY waste_classification waste_classification
COPY requirements.txt requirements.txt
COPY models models

RUN pip install --upgrade pip
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

CMD uvicorn waste_classification.api.fast:app --host 0.0.0.0 --port $PORT
