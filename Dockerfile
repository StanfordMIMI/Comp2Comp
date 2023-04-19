FROM python:3.8
COPY . /Comp2Comp
WORKDIR /Comp2Comp
RUN pip install -e .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y