FROM python:3.8-slim

ADD requirements.txt /
RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN mkdir app/
ADD event_publisher.py app/
ADD metrics_pb2.py app/
WORKDIR  app/

ENTRYPOINT ["python", "event_publisher.py"]
