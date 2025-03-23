from base_llm import BaseLlm
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage


class Ollama(BaseLlm):

    def text_completion(self):
        template = """Question: {question}
        Answer: Let's think step by step."""
        prompt = ChatPromptTemplate.from_template(template)
        model = OllamaLLM(model=self.llm_request.get_model())
        chain = prompt | model
        return chain.invoke({"question": self.llm_request.get_user_msg()})
        
    
    def chat(self):
        llm = ChatOllama(
            model=self.llm_request.get_model(),
            temperature=self.llm_request.get_temp(),
            num_predict=self.llm_request.get_num_ctx(),
        )

        # messages = [
        #     (
        #         "system",
        #         "You are a helpful assistant that translates English to French. Translate the user sentence.",
        #     ),
        #     ("human", self.llm_request.get_user_msg()),
        # ]
        # ai_msg = llm.invoke(messages)
        # print(ai_msg.content)

        prompt = ChatPromptTemplate.from_messages(
            # [
            #     (
            #         "system",
            #         "You are a helpful assistant that translates {input_language} to {output_language}.",
            #     ),
            #     ("human", "{input}"),
            # ]
            [
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm
        ai_msg_chain = chain.invoke(
            {
                "input": self.llm_request.get_user_msg(),
            }
        )
        # print(ai_msg_chain.content)

        # return ai_msg.content + " ||| " + ai_msg_chain.content
        return ai_msg_chain.content
    
    def embeddings(self):
        embed = OllamaEmbeddings(
            model=self.llm_request.get_model()
        )

        input_text = self.llm_request.get_user_msg()
        vector = embed.embed_query(input_text)
        # print(vector[:3])

        input_texts = [self.llm_request.get_user_msg(), self.llm_request.get_user_msg()]
        vectors = embed.embed_documents(input_texts)
        # print(len(vectors))
        # print(vectors[0][:3])

        return "generated embeddings: " + str(vector[:3])
    
    def train(self):
        modelfile='''
            FROM llama3.2
            SYSTEM You are mario from super mario bros.
            '''

        ollama.create(model='example', modelfile=modelfile)
