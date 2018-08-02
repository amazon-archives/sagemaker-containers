#!/usr/bin/env bash

# Get the account number associated with the current IAM credentials
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
REGION=$(aws configure get region)
REGION=${REGION:-us-west-2}

BUCKET_NAME="sagemaker-${REGION}-${AWS_ACCOUNT}"


aws s3 cp --recursive data "s3://${BUCKET_NAME}/DEMO-scikit-byo-sagemaker-containers/"