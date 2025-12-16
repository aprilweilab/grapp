# Example to build:
#   docker build . -t grapp:latest
# Example to run:
#   docker run -v $PWD:/working -it grapp:latest grapp filter -r 0-10000 /working/input.grg /working/output.grg

FROM ubuntu:22.04

RUN apt update && \
    apt install -y python3 python3-setuptools python3-pip python-is-python3 lz4 time && \
    pip install pygrgl && \
    pip install igdtools

COPY . /grapp
RUN pip install /grapp
