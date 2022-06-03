# start by pulling the python image
FROM python:3.8-slim-buster

WORKDIR /python-docker

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .
# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]

CMD ["main.py" ]