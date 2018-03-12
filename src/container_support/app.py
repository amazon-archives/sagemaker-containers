import inspect
import logging
import optparse
import os
import signal
import subprocess
import sys
import traceback
from pprint import pprint

import container_support.environment as env
from container_support.serving import Server

PYTHONPATH = 'PYTHONPATH'

SAGEMAKER_CALLER_FILE = 'SAGEMAKER_CALLER_FILE'

FRAMEWORK_TRAIN_PARAMETERS = ['user_module', 'training_environment']

logger = logging.getLogger(__name__)


class ServingEngine(object):
    def __init__(self):
        self.environment = None
        self._predict_fn = None

    def _model_fn(self, mode_dir):
        return None

    def _input_fn(self, data, content_type):
        return data

    def _output_fn(self, prediction, accept):
        return prediction, accept

    def _transform_fn(self):
        pass

    def _load_dependencies(self):
        pass

    def model_fn(self):
        def decorator(_model_fn):
            self._model_fn = _model_fn
            return _model_fn

        return decorator

    def input_fn(self):
        def decorator(_input_fn):
            self._input_fn = _input_fn
            return _input_fn

        return decorator

    def predict_fn(self):
        def decorator(_predict_fn):
            self._predict_fn = _predict_fn
            return _predict_fn

        return decorator

    def output_fn(self):
        def decorator(_output_fn):
            self._output_fn = _output_fn
            return _output_fn

        return decorator

    def transform_fn(self):
        def decorator(_transform_fn):
            self._transform_fn = _transform_fn
            return _transform_fn

        return decorator

    def load_dependencies(self):
        def decorator(_load_dependencies):
            self._load_dependencies = _load_dependencies
            return _load_dependencies

        return decorator

    def fork(self, environ, start_response):
        hosting_environment = env.HostingEnvironment()

        env.configure_logging()
        logger.info("creating Server instance")
        logger.info("importing user module")
        user_module = hosting_environment.import_user_module() if hosting_environment.user_script_name else None

        # load model
        has_model_fn = hasattr(user_module, 'model_fn')
        model_dir = hosting_environment.model_dir
        model = user_module.model_fn(model_dir) if has_model_fn else self._model_fn(model_dir)

        # if user has supplied a transform_fn, we can use the constructor directly
        if hasattr(user_module, 'transform_fn'):

            self._transform_fn = user_module.transform_fn
        else:
            input_fn = self._input_fn if not hasattr(user_module, 'input_fn') else user_module.input_fn

            predict_fn = self._predict_fn if not hasattr(user_module, 'predict_fn') else user_module.predict_fn

            output_fn = self._output_fn if not hasattr(user_module, 'output_fn') else user_module.output_fn

            if predict_fn:
                def transform_fn(model, data, content_type, accept):
                    input_data = input_fn(data, content_type)
                    prediction = predict_fn(input_data, model)
                    output_data, accept = output_fn(prediction, accept)
                    return output_data, accept

                self._transform_fn = transform_fn

        print(self.__dict__)

        server = Server("model server", self._transform_fn, model)
        logger.info("returning initialized server")
        return server.app(environ, start_response)

    def run(self):
        """Prepare the container for model serving, configure and launch the model server stack.
        """

        logger.info("reading config")
        hosting_environment = env.HostingEnvironment()
        hosting_environment.start_metrics_if_enabled()

        if hosting_environment.user_script_name:
            hosting_environment.download_user_module()

        logger.info('loading framework-specific dependencies')

        self._load_dependencies()

        nginx_pid = 0
        gunicorn_bind_address = '0.0.0.0:8080'
        if hosting_environment.use_nginx:
            logger.info("starting nginx")
            subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
            subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])
            gunicorn_bind_address = 'unix:/tmp/gunicorn.sock'
            nginx_pid = subprocess.Popen(['nginx', '-c', '/usr/local/etc/nginx.conf']).pid

        logger.info("starting gunicorn")
        module_app = os.environ[SAGEMAKER_CALLER_FILE]
        logger.info(module_app)

        gunicorn_pid = subprocess.Popen(["gunicorn",
                                         "--timeout", str(hosting_environment.model_server_timeout),
                                         "-k", "gevent",
                                         "-b", gunicorn_bind_address,
                                         "--worker-connections", str(1000 * hosting_environment.model_server_workers),
                                         "-w", str(hosting_environment.model_server_workers),
                                         "--log-level", "debug",
                                         module_app]).pid

        signal.signal(signal.SIGTERM, lambda a, b: self._sigterm_handler(nginx_pid, gunicorn_pid))

        children = {nginx_pid, gunicorn_pid} if nginx_pid else gunicorn_pid
        logger.info("inference server started. waiting on processes: %s" % children)

        while True:
            pid, _ = os.wait()
            if pid in children:
                break

        self._sigterm_handler(nginx_pid, gunicorn_pid)

    def _sigterm_handler(self, nginx_pid, gunicorn_pid):
        logger.info("stopping inference server")

        if nginx_pid:
            try:
                os.kill(nginx_pid, signal.SIGQUIT)
            except OSError:
                pass

        try:
            os.kill(gunicorn_pid, signal.SIGTERM)
        except OSError:
            pass

        sys.exit(0)


class TrainingEngine(object):
    def __int__(self):
        self.train_fn = None
        self.environment = None
        self.training_parameters = None

    def train(self, hyperparameters=None, env_vars=None, training_args=None):
        def decorator(train_fn):
            self.train_fn = train_fn
            self.add_hyperparameters(hyperparameters)
            self.add_env_vars(env_vars)
            self.add_training_args(training_args)
            return train_fn

        return decorator

    def add_hyperparameters(self, hyperparameters):
        pass

    def add_env_vars(self, env_vars):
        pass

    def add_training_args(self, training_args):
        pass

    def run(self):
        training_environment = env.TrainingEnvironment()
        logger.info("started training: {}".format(repr(self.__dict__)))

        try:
            training_environment.start_metrics_if_enabled()

            arg_spec = inspect.getargspec(self.train_fn)
            if arg_spec.args == FRAMEWORK_TRAIN_PARAMETERS:
                training_environment.download_user_module()
                user_module = training_environment.import_user_module()
                training_environment.load_training_parameters(user_module.train)
                self.train_fn(user_module, training_environment)
            else:
                training_parameters = training_environment.matching_parameters(self.train_fn)
                self.train_fn(**training_parameters)

            training_environment.write_success_file()
        except Exception as e:
            trc = traceback.format_exc()
            message = 'uncaught exception during training: {}\n{}\n'.format(e, trc)
            training_environment.write_failure_file(message, training_environment.base_dir)
            raise e


class ContainerSupport(object):
    def __init__(self):
        self.training_engine = TrainingEngine()
        self.serving_engine = ServingEngine()

    def __call__(self, environ, start_response):
        return self.serving_engine.fork(environ, start_response)

    def register_engine(self, engine):
        _type = type(engine)
        if _type is TrainingEngine:
            self.training_engine = engine
        elif _type is ServingEngine:
            self.serving_engine = engine
        else:
            raise ValueError('Type: {} is not a valid engine type', _type)

    def run(self):
        previous_frame = inspect.currentframe().f_back
        app_name, module_dir, module_name = self._get_caller_info(previous_frame)
        self._set_env_vars(app_name, module_dir, module_name)

        try:
            env.configure_logging()
            logging.info("running container entrypoint")

            parser = optparse.OptionParser()
            (options, args) = parser.parse_args()

            modes = {"train": self.training_engine.run, "serve": self.serving_engine.run}

            if len(args) != 1 or args[0] not in modes:
                raise ValueError("Illegal arguments: %s" % args)

            mode = args[0]
            logging.info("starting %s task", mode)

            modes[mode]()
        except Exception as e:
            trc = traceback.format_exc()
            message = 'uncaught exception: {}\n{}\n'.format(e, trc)
            logger.error(message)

    def _set_env_vars(self, app_name, module_dir, module_name):
        os.environ[PYTHONPATH] = '{}:{}'.format(module_dir, os.environ.get(PYTHONPATH))
        os.environ[SAGEMAKER_CALLER_FILE] = '{}:{}'.format(module_name, app_name)

    def _get_caller_info(self, previous_frame):
        (filename, line_number, function_name, lines, index) = inspect.getframeinfo(previous_frame)
        method_name = '.run()'
        app_name = ''.join(lines[0].split()).replace(method_name, '')
        module_dir = os.path.dirname(filename)
        module_name = os.path.basename(filename)[:-3]
        return app_name, module_dir, module_name
