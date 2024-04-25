import os
from dotenv import load_dotenv
from typing import List, Any
from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import (
    CHUNK_SIZE_SPLIT,
    CHUNK_OVERLAP,
    DOCUMENTS_PATH,
    LLM_MODEL,
    SEARCH_ITEMS
)

class VectorStore:
    def __init__(self, base_path: str, chunk_size: int, chunk_overlap: int, model_name: str, k_items: int):
        self.base_path = base_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.k_items = k_items

    def load_documents(self) -> List[str]:
        """
        Loads documents from the specified directory
        """
        loader = DirectoryLoader(self.base_path, glob="*.md", loader_cls=UnstructuredMarkdownLoader)
        return loader.load()

    def split_documents(self, docs: List[str]) -> List[str]:
        """
        Splits documents into chunks based on the specified chunk size and overlap.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def get_embedding_model(self) -> Any:
        """
        Retrieves the appropriate embedding model based on the model name.
        """
        if self.model_name.startswith("gpt"):
            return OpenAIEmbeddings()
        else:
            return None  # OllamaEmbeddings()

    def build_retriever(self, splits: List[str], embedding_model: Any) -> Any:
        """
        Builds a document retriever based on document splits and the embedding model.
        """
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
        return vectorstore.as_retriever(search_kwargs={"k": self.k_items})

    def get_retriever(self) -> Any:
        """
        Creates and returns a fully configured document retriever.
        """
        docs = self.load_documents()
        splits = self.split_documents(docs)
        embedding_model = self.get_embedding_model()
        retriever = self.build_retriever(splits, embedding_model)
        return retriever

if __name__ == "__main__":

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    vector_store = VectorStore(DOCUMENTS_PATH, CHUNK_SIZE_SPLIT, CHUNK_OVERLAP, LLM_MODEL, SEARCH_ITEMS)
    retriever = vector_store.get_retriever()

    test_question = "When sage maker create a model what happen?"
    response = retriever.invoke(test_question)
    print("Nro of docs: ", len(response))
    print("Source of first doc:", response[0].metadata["source"])
    print("Content: ", response[0].page_content[:100])
