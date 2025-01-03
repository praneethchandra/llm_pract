from base_llm import BaseLlm
import getpass
import os
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


class ChatGptLlm(BaseLlm):

    def __init__(self, llm_req):
        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")
        super().__init__(llm_req)

    def text_completion(self):
        template = """Question: {question}
        Answer: Let's think step by step."""
        prompt = ChatPromptTemplate.from_template(template)
        llm = OpenAI(
            model_name=self.llm_request.get_model()
        )
        chain = prompt | llm
        return chain.invoke({"question": self.llm_request.get_user_msg()})
    

    def chat(self):
        llm = ChatOpenAI(
            model=self.llm_request.get_model(),
            temperature=self.llm_request.get_temp(),
            # num_predict=self.llm_request.get_num_ctx(),
        )
    
        prompt = ChatPromptTemplate.from_messages(
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
        print(ai_msg_chain.content)

        return ai_msg_chain.content

    def embeddings(self):
        embed = OpenAIEmbeddings(
            model=self.llm_request.get_model()
        )

        input_text = self.llm_request.get_user_msg()
        vector = embed.embed_query(input_text)
        print(vector[:3])

        input_texts = [self.llm_request.get_user_msg(), self.llm_request.get_user_msg()]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        print(vectors[0][:3])

        return "generated embeddings: " + str(vector[:3])