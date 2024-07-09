import chromadb
import logging
import sys

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, PromptTemplate
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import TextNode

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global query_engine
query_engine = None

def init_llm():
    llm = Ollama(model="llama3", base_url = "http://127.0.0.1:11434", request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model


def init_index(embed_model):
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("iollama")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    node1 = TextNode(text="Петя любит музыку.", id_="0")
    node2 = TextNode(text="Пятя любит гулять полесу зимой.", id_="1")
    node3 = TextNode(text="Петя и Маша встречаются уже два месяца.", id_="2")
    node4 = TextNode(text="У Маши есть кошка Дуся и собака Тор.", id_="3")
    node5 = TextNode(text="Артем - сын Кати.", id_="4")
    node6 = TextNode(text="Маша любит делать все аккуратно.", id_="5")
    node7 = TextNode(text="Маша очень любит пиццу.", id_="6")
    node8 = TextNode(text="Петя любит футбол и пиво.", id_="7")
    node9 = TextNode(text="Маша не любит, когда Петя пьет пиво.", id_="8")
    node10 = TextNode(text="Петя и Маша постоянно ругаются.", id_="9")
    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10]
    index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)

    return index


def init_query_engine(index):
    template = (
        "Представь, что ты друг нашей семьи. Давно знаком с нами. Ты знаешь много информации о нас. Твоя задача предоставить точные, целостные ответы на вопросы о нашей сеье.\n\n"
        "Используй дополнительную информацию для ответов:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Принимая во внимание информацию выше, ответь на вопрос\n\n"
        "Question: {query_str}\n\n"         
    )
    qa_template = PromptTemplate(template)
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    return query_engine


def chat(input_question, user):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("got response from llm - %s", response)

    return response.response


def chat_cmd(query_engine):
    while True:
        input_question = input("Enter your question (or 'exit' to quit): ")
        if input_question.lower() == 'exit':
            break

        response = query_engine.query(input_question)
        logging.info("got response from llm - %s", response)


if __name__ == '__main__':
    init_llm()
    index = init_index(Settings.embed_model)
    q_engine = init_query_engine(index)
    chat_cmd(q_engine)
