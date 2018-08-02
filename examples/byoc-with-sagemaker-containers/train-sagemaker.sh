#!/usr/bin/env bash

IMAGE_NAME=scikit-learn-image

docker run -it -v $(pwd)/data:/opt/ml/input/data/training/ -v $(pwd)/model:/opt/ml/model/  ${IMAGE_NAME} train --batch-size 234 -v --gpu-only --learning-rate 1.2
