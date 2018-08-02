FROM python:2

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential nginx git

RUN pip install --upgrade \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    pandas


RUN  git clone https://github.com/mvsusp/sagemaker-containers.git -b mvs-sagemaker-containers-train-improvements && cd sagemaker-containers && pip install .

COPY decision_trees /decision_trees

ENV PYTHONPATH /decision_trees


ENV SAGEMAKER_TRAINING_MODULE train
ENV SAGEMAKER_SERVING_MODULE serve:main
