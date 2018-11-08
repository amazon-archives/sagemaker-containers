# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import os
import re
import signal
import subprocess

import pkg_resources

import sagemaker_containers
from sagemaker_containers import _env, _files, _logging

logger = _logging.get_logger()

UNIX_SOCKET_BIND = 'unix:/tmp/gunicorn.sock'
PORT = os.getenv("SAGEMAKER_BIND_TO_PORT", "8080")
HTTP_BIND = '0.0.0.0:{}'.format(PORT)

nginx_config_file = pkg_resources.resource_filename(sagemaker_containers.__name__, '/etc/nginx.conf')
nginx_config_template_file = pkg_resources.resource_filename(sagemaker_containers.__name__, '/etc/nginx.conf.template')


def _create_nginx_config():
    template = _files.read_file(nginx_config_template_file)

    pattern = re.compile(r'%(\w+)%')
    template_values = {
        'NGINX_HTTP_PORT': PORT
    }

    config = pattern.sub(lambda x: template_values[x.group(1)], template)

    logger.info('nginx config: \n%s\n', config)

    _files.write_file(nginx_config_file, config)


def _add_sigterm_handler(nginx, gunicorn):
    def _terminate(signo, frame):
        if nginx:
            try:
                os.kill(nginx.pid, signal.SIGQUIT)
            except OSError:
                pass

        try:
            os.kill(gunicorn.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)


def start(module_app):
    env = _env.ServingEnv()
    gunicorn_bind_address = HTTP_BIND

    nginx = None

    if env.use_nginx:
        gunicorn_bind_address = UNIX_SOCKET_BIND
        _create_nginx_config()
        nginx = subprocess.Popen(['nginx', '-c', nginx_config_file])

    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(env.model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', gunicorn_bind_address,
                                 '--worker-connections', str(1000 * env.model_server_workers),
                                 '-w', str(env.model_server_workers),
                                 '--log-level', 'info',
                                 module_app])

    _add_sigterm_handler(nginx, gunicorn)

    # wait for child processes. if either exit, so do we.
    pids = {c.pid for c in [nginx, gunicorn] if c}
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
