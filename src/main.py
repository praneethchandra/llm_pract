from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0.8,
    num_predict=256,
)

messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming.")
]

# invoke
print("====> RESPONSE after calling plain invoke: " + llm.invoke(messages).content)

# stream
messages = [
    ("human", "Return the words Hello World!"),
]
stream = llm.stream(messages)
for chunk in stream:
    print(chunk)

# stream append chunks
stream = llm.stream(messages)
full = next(stream)
for chunk in stream:
    full += chunk

print("====> RESPONSE after calling stream with appending: " + full.content)

'''
# async await
messages = [
    ("human", "Hello how are you!"),
]
await llm.ainvoke(messages)

# async astream
messages = [
    ("human", "Say hello world!"),
]
async for chunk in llm.astream(messages):
    print(chunk.content)

# async abatch
messages = [
    ("human", "Say hello world!"),
    ("human","Say goodbye world!")
]
await llm.abatch(messages)
'''


# return response in JSON format
json_llm = ChatOllama(
    model="llama3.2",
    temperature=0.8,
    num_predict=256,
    format="json",
    )
messages = [
    ("human", "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. Respond using JSON only."),
]
print("====> RESPONSE after calling invoke with output format as JSON: " + json_llm.invoke(messages).content)