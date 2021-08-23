# Docker Microservices Assignment

## Learning Outcomes
- Understanding the microservies design paradigm and how it is useful in large scale applications
- Understanding the concept of containerization
- Understading how to use docker and docker-compose for container orchestration
- Understanding HTTP requests and how containers interact with each other
- Understading how to scale containers for better load management


## Dataset
Wine Predicting dataset (tentative)

## Microservices

### processr
This service will take in raw data and preprocess it. After the preprocessing is done, it will send the cleaned data to `trainr` as a POST request where training will take place.
Routes:
- POST `/api/process`: It will take a list of data points in JSON format

### trainr
This service will take in cleaned data and train an ML model over it. The trained model will be stored in a common mounted volume so that it can be further used.
Routes:
- POST `/api/train`: It will take a list of cleaned data points 

### predictr
This service will read the pre-trained model from the common volume mount and use it to make predictions. This will be used by the consumer.
Routes:
- POST `/api/predict`: It will take a list of parameters and return a list of predicted labels

## Docker Compose

The docker-compose file for stack will look like this
```yaml
version: '3'

services:
  processr:
    deploy:
      replicas: 1
    build: processr
    environment:
      - TRAINR_ENDPOINT=http://trainr:7777
    ports:
      - "8888:8888"

  trainr:
    deploy:
      replicas: 1
    build: trainr
    environment:
      - PREDICTR_ENDPOINT=http://predictr:9999
    ports:
      - "7777:7777"
    volumes:
      - ./models:/app/models

  predictr:
    deploy:
      replicas: 1
    build: predictr
    ports:
      - "9999:9999"
    volumes:
      - ./models:/app/models
```
