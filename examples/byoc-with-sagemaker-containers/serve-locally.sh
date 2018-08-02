#!/usr/bin/env bash

IMAGE_NAME=scikit-learn-image

docker run -p 8080:8080 -v $(pwd)/model:/opt/ml/model/  ${IMAGE_NAME} serve
