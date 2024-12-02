class LLMRequest:
    _model: str = "llama3.2"
    _purpose: str = "chat"
    _temp: float = 0.8
    _num_ctx: int = 2048
    _topk: int = 40
    _format: str = ""

    def __init__(self, model, purpose, temp, num_ctx, topk, format):
        self._model = model
        self._purpose = purpose
        self._temp = temp
        self._num_ctx = num_ctx
        self._topk = topk
        self._format = format

    def get_model(self):
        return self._model
    
    def get_purpose(self):
        return self._purpose
    
    def get_temp(self):
        return self._temp
    
    def get_num_ctx(self):
        return self._num_ctx
    
    def get_topk(self):
        return self._topk

    def get_format(self):
        return self._format