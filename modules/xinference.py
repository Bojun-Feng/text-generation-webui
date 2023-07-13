'''
Based on
https://github.com/abetlen/llama-cpp-python

Documentation:
https://abetlen.github.io/llama-cpp-python/
'''

from xinference.client import Client


class XinferenceModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(
            self,
            endpoint,
            model_uid,
            model_name,
            model_size,
            quantization,
    ):
        assert isinstance(endpoint, str)
        self.endpoint = endpoint
        client = Client(self.endpoint)

        if model_uid == '0':
            model_uid = client.launch_model(
                model_name=model_name,
                model_size_in_billions=int(model_size),
                quantization=quantization,
            )
        self.model_uid = model_uid
        self.model = client.get_model(self.model_uid)
        return self, self
