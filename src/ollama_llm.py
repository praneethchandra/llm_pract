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
        model = OllamaLLM(model="llama3")
        chain = prompt | model
        return chain.invoke({"question": self.llm_request.get_user_msg()})
        
    
    def chat(self):
        llm = ChatOllama(
            model="llama3.2",
            temperature=0.8,
            num_predict=256,
        )

        print("Invocation!!")
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", self.llm_request.get_user_msg()),
        ]
        ai_msg = llm.invoke(messages)
        print(ai_msg.content)

        print("Chaining!!")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that translates {input_language} to {output_language}.",
                ),
                ("human", "{input}"),
            ]
        )

        chain = prompt | llm
        ai_msg_chain = chain.invoke(
            {
                "input_language": "English",
                "output_language": "German",
                "input": self.llm_request.get_user_msg(),
            }
        )
        print(ai_msg_chain.content)

        return ai_msg.content + " ||| " + ai_msg_chain.content
    
    def embeddings(self):
        embed = OllamaEmbeddings(
            model="llama3.2"
        )

        input_text = self.llm_request.get_user_msg()
        vector = embed.embed_query(input_text)
        print(vector[:3])

        input_texts = [self.llm_request.get_user_msg(), self.llm_request.get_user_msg()]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        print(vectors[0][:3])

        return "generated embeddings: " + str(vector[:3])
