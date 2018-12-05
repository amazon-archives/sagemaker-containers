import time

import pytest

import sagemaker_containers
from sagemaker_containers import _timeout


def test_timeout():
    sec = 2
    with pytest.raises(sagemaker_containers._timeout.TimeoutError):
        with _timeout.timeout(seconds=sec):
            print("Waiting and testing timeout, it should happen in {} seconds.".format(sec))
            time.sleep(sec + 1)
