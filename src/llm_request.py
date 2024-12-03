class LLMRequest:
    _model: str = "llama3.2"
    _purpose: str = "chat"
    _temp: float = 0.8
    _num_ctx: int = 2048
    _topk: int = 40
    _format: str = ""
    _user_msg: str
    _user_msg_history: list

    def __init__(self, model, purpose, temp, num_ctx, topk, format):
        self._model = model
        self._purpose = purpose
        self._temp = temp
        self._num_ctx = num_ctx
        self._topk = topk
        self._format = format

    def get_model(self) -> str:
        return self._model
    
    def get_purpose(self) -> str:
        return self._purpose
    
    def get_temp(self) -> float:
        return self._temp
    
    def get_num_ctx(self) -> int:
        return self._num_ctx
    
    def get_topk(self) -> int:
        return self._topk

    def get_format(self) -> str:
        return self._format
    
    def set_user_msg(self, msg: str):
        self._user_msg = msg
    
    def get_user_msg(self) -> str:
        return self._user_msg
    
    def set_user_msg_history(self, msg_history: list):
        self._user_msg_history = msg_history
    
    def get_user_msg_history(self) -> list:
        return self._user_msg_history