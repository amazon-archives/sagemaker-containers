![alt text](branding/icon/sagemaker-banner.png "Logo Title Text 1")

# Amazon SageMaker Containers

**SageMaker Containers** is the open source library that the SageMaker team uses to create the open source containers
that run on [Amazon SageMaker](https://aws.amazon.com/documentation/sagemaker/):

- https://github.com/aws/sagemaker-tensorflow-containers
- https://github.com/aws/sagemaker-mxnet-containers
    
 You can use SageMaker Containers as a **library** with helper functions for your training script or to 
 **create** your own ML framework container compatible with SageMaker. 
 
## Install
 
### Installing from source
 ```bash
pip install -U .
```

## Getting started
### Using it as a **library**
```sagemaker_containers.training_environment``` contains information about the training container including: 
hyperparameters, instance information (number of CPUs, GPUs, host name), input data information, environment
variables, etc.

Let's suppose that you want to train the Keras script below in SageMaker:
```my_training_script.py```
```python
import argparse
import os

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--training-data-dir', type=str)
parser.add_argument('--batch-size', type=int)
parser.add_argument('--model-dir', type=str)

args = parser.parse_args()

data = np.load(os.path.join(args.training_data_dir, 'training_data.npz'))
x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=20))
model.add(tf.keras.layers.Dropout(0.5))

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=args.batch_size)

# saves the model in the end of training
model.save(os.path.join(args.model_dir, 'saved_model.h5'))
```

To be able to train the script above in SageMaker, you need to provide the parameters ```--training-data-dir```,
```--batch-size```, ```--model-dir```. ```training_environment``` makes this process very easy:

```python
import sagemaker_containers.training_environment as env
...

# TODO (mvsusp) - create env.training_data_dir and env.evaluation_data_dir helpers
parser.add_argument('--training-data-dir', type=str, default=env.training_data_dir)
parser.add_argument('--batch-size', type=int, default=env.batch_size)
parser.add_argument('--model-dir', type=str, default=env.model_dir)
```

We can use [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) to submit ```my_training_script.py``` to
trainining using the ```TensorFlow``` estimator:

```python
# TODO (mvsusp) - allow sagemaker-python-sdk estimator to work in the scenario below
from sagemaker.tensorflow import TensorFlow

estimator = TensorFlow('train.py', role='SageMakerRole', 
                       train_instance_type='ml.p2.xlarge', train_instance_count = 1)
estimator.fit({'training': 's3://my/bucket/with/the/training/data'})
```

### Creating a new ML framework container compatible with SageMaker Training
The core training structure of SageMaker Containers is ```Training```. When training starts, e.g. 
`training.run()`, the `Training` instance executes the following steps:

1. install any required python dependencies
2. start customs metrics that will be reported to CloudWatch Logs
3. load SageMaker Training environment information, including default configuration files, directories, 
    environment variables, and hyperparameters
4. download the user script (or Python package) containing the functions required by the framework
5. start the training process
6. report success/failure
  
Let's suppose that we want to create a Keras Container that trains any Keras model provided by the **user script**.

Let's suppose that to be able to use our Containers, users need to submit a Python script containing a function with 
the following signature:  
```python
def train(training_dir, hyperparameters):
    """Return a Keras model that will be trained by our SageMaker Keras container.
    
    Args:
        - training_dir (string): the path to the directory containing the training 
            data provided by the user.
        - hyperparameters (dict[string, object]): a map containing the hyperparameters 
            provided by the user.
         
    Returns:
        - (Keras.Model): the model that will be saved in the end of the training.
    """
```

#### Implementing our Keras Framework Container
```Training``` will install the user provided script as a module, and prepare the training environment for us. All we 
have to do is implement a function, ```training_process_fn```, which implements the training process of our framework.

```training_process_fn``` takes as parameters **user module** and the **training environment**, and is responsible to 
save the trained model in end of training:
```python
import os

from sagemaker_containers import Training

def keras_framework_training_process(user_module, training_environment):
    """This function implements our Keras Framework.
    
    Args:
        - user_module (module): the python script provided by the user loaded as a Python ```module```
        - training_environment (sagemaker_containers.TrainingEnvironment): the container's 
            training environment
    """
    # retrieves the training dir from training channels
    training_dir = training_environment.channel_input_dirs['training']
    
    # retrieves the hyperparameters submitted by the user
    hps = training_environment.hyperparameters
    
    # invokes user's ```train``` function
    model = user_module.train(training_dir, hps)
    
    # saves the model in the end of training
    model.save(os.path.join(training_environment.model_dir, 'saved_model.h5'))

# Creates the training from the training process function and start training.
Training.from_train_fn(train_fn=keras_framework_training_process).run()
```

#TODO (mvsusp) - complete the example after implementing support for Docker containers.

#### An example of user provided script for training:
__train.py__
```python
import os

import keras
import numpy as np

def train(training_dir, hyperparameters):
    data = np.load(os.path.join(training_dir, hyperparameters['training_data_file']))
    x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_dim=20))
    model.add(keras.layers.Dropout(0.5))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=128)
    return model
```
#### Submitting a Python script for training using our Keras container
We can use [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) to submit a scripts to our 
containers:

```python
# TODO (mvsusp) - allow sagemaker-python-sdk estimator to work in the scenario below
from sagemaker.estimator import Estimator

estimator = Estimator('train.py', role='SageMakerRole', 
                      train_instance_type='ml.p2.xlarge', train_instance_count = 1)
estimator.fit({'training': 's3://my/bucket/with/the/training/data'})
```

You can look at more information on how to use ```SageMaker Python SDK``` here: https://github.com/aws/sagemaker-python-sdk#byo-docker-containers-with-sagemaker-estimators.

