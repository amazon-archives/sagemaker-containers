#!/usr/bin/env bash

IMAGE_NAME=scikit-learn-image

docker run -it -v $(pwd)/data:/opt/ml/input/data/training/ -v $(pwd)/model:/opt/ml/model/  ${IMAGE_NAME} train --max-leaf-nodes 10
