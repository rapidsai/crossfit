class Model:
    def __init__(self, path_or_name: str, max_mem_gb: int = 16):
        self.path_or_name = path_or_name
        self.max_mem_gb = max_mem_gb

    def load_model(self, device="cuda"):
        raise NotImplementedError()

    def load_tokenizer(self):
        raise NotImplementedError()

    def load_on_worker(self, worker):
        raise NotImplementedError()

    def call_on_worker(self, worker, *args, **kwargs):
        return worker.torch_model(*args, **kwargs)

    def get_model(self, worker):
        return worker.torch_model

    def estimate_memory(self, max_num_tokens: int, batch_size: int) -> int:
        raise NotImplementedError()

    def max_seq_length(self) -> int:
        raise NotImplementedError()
