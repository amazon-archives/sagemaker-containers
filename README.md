# SageMaker Containers

SageMaker Containers contains common functionality necessary to create a container compatible with SageMaker. It can be simply used by any container by just installing the module:

```bash
pip install sagemaker-containers
```

## Getting Started -  How an user script is executed for training in SageMaker

The objective of this tutorial is to explain how a script is executed inside any container using **SageMaker Containeirs**.

### Creating the training job

A SageMaker training job created using [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk#sagemaker-python-sdk-overview) takes an user script containing the model to be trained, the Hyperparameters required by the script, and information about the input data. For example:

```python

# for complete list of parameters, see 
# https://github.com/aws/sagemaker-python-sdk#sagemaker-python-sdk-overview
estimator = Chainer(entry_point='user-script.py', 
                    hyperparameters={'batch-size':256, 'learning-rate':0.0001, communicator:'pure_nccl'},
                    ...) 

# starts the training job with an input data channel named training pointing to s3://bucket/path/to/training/data
# for more information about data channels, see
# https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig
chainer_estimator.fit({'training': 's3://bucket/path/to/training/data', 'testing': 's3://bucket/path/to/testing/data')
```

### How a script is executed inside the container

When the container starts for training, **SageMaker Containers** installs the user script as a Python module. The module name matches the script name. In the case above, **user-script.py** is transformed in a Python module named **user-script**.

After that, the Python interpreter executes the user module, passing ```hyperparameters``` as script arguments. The example above will be executed by **SageMaker Containers** as follow:

```python
python -m user-script --batch-size 256 --learning_rate 0.0001 --communicator pure_nccl
```

An user provide script consumes the hyperparameters using any argument parsing library, for example:

```python
  if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--learning-rate', type=int, default=1)
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--communicator', type=str)
  parser.add_argument('--frequency', type=int, default=20)

  args = parser.parse_args()
  ...
```

### Reading additional information from the container

Very often, an user script needs additional information from the container that is not available in ```hyperparameters```.
SageMaker Containers writes this information as **environment variables** that are available inside the script.

For example, the example above can read information about the **training** channel provided in the training job request:

```python
  if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  ...
  
  # reads input channels training and testing from the environment variables
  parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
  parser.add_argument('--testing', type=str, default=os.environ['SM_CHANNEL_TESTING'])

  args = parser.parse_args()
  ...
```
### List of provided environment variables by SageMaker Containers

The list of the environment variables is logged and available in cloudwatch logs. From the example above:
```json
SM_NUM_GPUS=1
SM_NUM_CPUS=4
SM_NETWORK_INTERFACE_NAME=ethwe

SM_CURRENT_HOST=algo-1
SM_HOSTS=["algo-1","algo-2"]
SM_LOG_LEVEL=20

SM_USER_ARGS=["--batch-size","256","--learning_rate","0.0001","--communicator","pure_nccl"]

SM_HP_LEARNING_RATE=0.0001
SM_HP_BATCH-SIZE=10000

SM_HPS=
{
    "batch-size": 10000,
    "epochs": 1
}

SM_CHANNELS=["testing","training"]
SM_CHANNEL_TRAINING=/opt/ml/input/data/training
SM_CHANNEL_TESTING=/opt/ml/input/data/test

SM_MODULE_NAME=distributed_customer_script
SM_MODULE_DIR=s3://sagemaker-{aws-region}-{aws-id}/{training-job-name}/source/sourcedir.tar.gz

SM_INPUT_DIR=/opt/ml/input
SM_INPUT_CONFIG_DIR=/opt/ml/input/config
SM_OUTPUT_DIR=/opt/ml/output
SM_OUTPUT_DATA_DIR=/opt/ml/output/data/algo-1
SM_MODEL_DIR=/opt/ml/model

SM_RESOURCE_CONFIG=
{
    "current_host": "algo-1",
    "hosts": [
        "algo-1",
        "algo-2"
    ]
}

SM_INPUT_DATA_CONFIG=
{
    "test": {
        "RecordWrapperType": "None",
        "S3DistributionType": "FullyReplicated",
        "TrainingInputMode": "File"
    },
    "train": {
        "RecordWrapperType": "None",
        "S3DistributionType": "FullyReplicated",
        "TrainingInputMode": "File"
    }
}


SM_FRAMEWORK_MODULE=sagemaker_chainer_container.training:main

SM_TRAINING_ENV=
{
    "channel_input_dirs": {
        "test": "/opt/ml/input/data/testing",
        "train": "/opt/ml/input/data/training"
    },
    "current_host": "algo-1",
    "framework_module": "sagemaker_chainer_container.training:main",
    "hosts": [
        "algo-1",
        "algo-2"
    ],
    "hyperparameters": {
        "batch-size": 10000,
        "epochs": 1
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "test": {
            "RecordWrapperType": "None",
            "S3DistributionType": "FullyReplicated",
            "TrainingInputMode": "File"
        },
        "train": {
            "RecordWrapperType": "None",
            "S3DistributionType": "FullyReplicated",
            "TrainingInputMode": "File"
        }
    },
    "input_dir": "/opt/ml/input",
    "job_name": "preprod-chainer-2018-05-31-06-27-15-511",
    "log_level": 20,
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-{aws-region}-{aws-id}/{training-job-name}/source/sourcedir.tar.gz",
    "module_name": "distributed_customer_script",
    "network_interface_name": "ethwe",
    "num_cpus": 4,
    "num_gpus": 1,
    "output_data_dir": "/opt/ml/output/data/algo-1",
    "output_dir": "/opt/ml/output",
    "resource_config": {
        "current_host": "algo-1",
        "hosts": [
            "algo-1",
            "algo-2"
        ]
    }
}
```
## Environment Variables full specification:

#### SM_NUM_GPUS
The number of gpus available in the current container.

##### Usage example
###### log
```json
SM_NUM_GPUS=1
```
###### script (arg parse)
```python
parser.add_argument('num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
```
###### script (as variable)
```python
num_gpus = int(os.environ['SM_NUM_GPUS'])
```


#### SM_NUM_CPUS
The number of cpus available in the current container.

##### Usage example
###### log
```json
SM_NUM_CPUS=32
```
###### script (arg parse)
```python
parser.add_argument('num_cpus', type=int, default=os.environ['SM_NUM_CPUS'])
```
###### script (as variable)
```python
num_cpus = int(os.environ['SM_NUM_CPUS'])
```

### SM_NETWORK_INTERFACE_NAME
Name of the network interface, useful for distributed training.

#### Usage example
###### log
```json
SM_NETWORK_INTERFACE_NAME=ethwe
```
###### script (arg parse)
```python
parser.add_argument('network_interface', type=str, default=os.environ['SM_NETWORK_INTERFACE_NAME'])
```
###### script (as variable)
```python
network_interface = os.environ['SM_NETWORK_INTERFACE_NAME']
```


## License

This library is licensed under the Apache 2.0 License. 
