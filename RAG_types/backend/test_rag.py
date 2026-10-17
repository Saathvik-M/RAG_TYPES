from rag.naive import NaiveRAG

rag = NaiveRAG()

response = rag.ask(

    "What is Artificial Intelligence?"

)

print()

print(response["answer"])