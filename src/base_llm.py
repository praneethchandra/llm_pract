
class BaseLlm:
    llm_request = None
    def __init__(self, llm_req):
        self.llm_request = llm_req

    def text_completion(self):
        print("Base implementation not exist!")
        pass

    def chat(self):
        print("Base implementation not exist!")
        pass

    def embeddings(self):
        print("Base implementation not exist!")
        pass

    def invoke(self):
        if self.llm_request.get_purpose() == "Text Completion":
            return self.text_completion()
        elif self.llm_request.get_purpose() == "Chat":
            return self.chat()
        elif self.llm_request.get_purpose() == "Embeddings":
            return self.embeddings()