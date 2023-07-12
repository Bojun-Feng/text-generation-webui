'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

import re
from functools import partial

import xinference
from xinference.client import Client

from modules import shared
from modules.callbacks import Iteratorize
from modules.logging_colors import logger


class XinferenceModel:
    def __init__(self):
        pass
        # try:
        #     import xinference
        #     from xinference.client import Client
        # except ImportError:
        #     error_message = "Failed to import module 'xinference'"
        #     installation_guide = [
        #         "Please make sure 'xinference' is installed properly.\n",
        #         "You can visit the original git repo for latest installation instructions:\n",
        #         "https://github.com/xorbitsai/inference",
        #     ]
        #
        #     raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    @classmethod
    def from_pretrained(self, path, endpoint, model_uid):
        self.endpoint = endpoint
        self.model_uid = model_uid

        client = Client(f"http://localhost:{self.endpoint}")
        self.model = client.get_model(self.model_uid)
        return None

    def encode(self, string, **kwargs):
        return self.tokenizer.encode(string)

    def decode(self, string, **kwargs):
        return self.tokenizer.decode(string)[0]

    def generate(self, prompt, state, callback=None):
        pass
        # prompt = prompt if type(prompt) is str else prompt.decode()
        # return self.model.generate(
        #     prompt=prompt
        # )

    def generate_with_streaming(self, *args, **kwargs):
        pass
        # prompt = prompt if type(prompt) is str else prompt.decode()
        # return self.model.generate(
        #     prompt=prompt
        # )
