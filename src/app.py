import streamlit as st
from ollama_llm import Ollama
from llm_request import LLMRequest
from chatgpt_llm import ChatGptLlm

llmrequest = None

with st.sidebar:
    
    # select box to choose which framework to use for LLM models
    framework_llm = st.sidebar.selectbox(
        "What framework to use?",
        ('ollama',
         'openai',
        'huggingface',
        'other')
    )

    model_name = None

    if framework_llm == "ollama":
        model_name = st.selectbox(
           "which model to use?",
            ("llama3.2", "bespoke-minicheck", "all-minillm", "codellama", "meditron" )
        )
    elif framework_llm == "huggingface":
        pass
    elif framework_llm == "openai":
        model_name = st.selectbox(
           "which model to use?",
            ("gpt-4o-mini", "text-embedding-3-large", "got-4o", "gpt-4o-audio-preview", "gpt-3.5-turbo-instruct" )
        )
    else:
        pass



    purpose = st.selectbox(
        "Purpose?",
        ("Text Completion", "Chat", "Embeddings")
    )

    temp_slider = st.sidebar.slider(
        'Select temperature',
        0.0, 1.0, 0.8
    )

    num_ctx = st.sidebar.slider(
        'Select size of context window',
        0, 4096, 2048
    )

    top_k = st.sidebar.slider(
        "Probability of response to be conservative",
        0, 100, 10
    )

    format_of_response = st.selectbox(
        "Format of response?",
        ("text", "json")
    )

    llmrequest = LLMRequest(model_name, purpose, temp_slider, num_ctx, top_k, format_of_response)

st.title(":cloud: LLM Interface")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    llmrequest.set_user_msg(prompt)

    if framework_llm == "ollama":
        ollamallm = Ollama(llmrequest)
        asst_response = ollamallm.invoke()
        st.session_state.messages.append({"role": "assistant", "content": asst_response})
        st.chat_message("assistant").write(asst_response)

    elif framework_llm == "openai":
        openaillm = ChatGptLlm(llmrequest)
        asst_response = openaillm.invoke()
        st.session_state.messages.append({"role": "assistant", "content": asst_response})
        st.chat_message("assistant").write(asst_response)
