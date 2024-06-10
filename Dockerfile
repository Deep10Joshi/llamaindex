# pull official base image
FROM python:3.10-slim

# set work directory
WORKDIR /app

COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY . .

# set environment variables
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host", "0.0.0.0"]