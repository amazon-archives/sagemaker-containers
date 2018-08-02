#!/usr/bin/env bash

IMAGE_NAME=scikit-learn-image

# Get the account number associated with the current IAM credentials
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
REGION=$(aws configure get region)
REGION=${REGION:-us-west-2}


ECR_IMAGE_NAME="${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${IMAGE_NAME}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${IMAGE_NAME}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${REGION} --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker tag ${IMAGE_NAME} ${ECR_IMAGE_NAME}

docker push ${ECR_IMAGE_NAME}
