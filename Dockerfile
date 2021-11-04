FROM ubuntu:20.04
LABEL description="This is a custom docker image for classification project"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt -y update
RUN apt -y install python3-pip
WORKDIR /project
ENV HYDRA_FULL_ERROR=1
ENV WANDB_MODE=online
COPY . .
RUN ls
RUN pip3 install -r requirements.txt
RUN pip3 install -Ue .
CMD ["python3", "classification_project/train.py"]
