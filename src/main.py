import os 
from vector_store import VectorStore
from dotenv import load_dotenv
from rags_prompts import PROMPT_RAG_TEMPLATE
from utils import format_docs
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import (
    DOCUMENTS_PATH,
    LLM_MODEL,
    TEMPERATURE,
    CHUNK_OVERLAP,
    CHUNK_SIZE_SPLIT,
    SEARCH_ITEMS
)

if __name__ == "__main__":

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    prompt = PromptTemplate.from_template(PROMPT_RAG_TEMPLATE)

    model = ChatOpenAI(api_key=OPENAI_API_KEY, model= LLM_MODEL, temperature=TEMPERATURE)

    retriever = VectorStore(
         base_path = DOCUMENTS_PATH,
         chunk_size=CHUNK_SIZE_SPLIT,
         chunk_overlap=CHUNK_OVERLAP,
         model_name=LLM_MODEL,
         k_items=SEARCH_ITEMS
    ).get_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    print(rag_chain.invoke("Whe sage maker create a model what happen?"))