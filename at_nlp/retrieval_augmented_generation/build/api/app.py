import uvicorn
import toml
import requests
import os

from langchain_mixtral import Mixtral8x7b
from fastapi import FastAPI
from loguru import logger as log
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.language_models.llms import BaseLLM

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from typing import Dict, List, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

app = FastAPI()


__API_HOST__ = os.environ.get("API_HOST", "127.0.0.1")
__API_PORT__ = os.environ.get("API_PORT", "5000")
__LLM_HOST__ = os.environ.get("LLM_SERVER_HOST", "127.0.0.1")
__LLM_PORT__ = os.environ.get("LLM_SERVER_PORT", "7000")


def format_docs(docs) -> None:
    return "\n\n".join(doc.page_content for doc in docs)


def llm_response(prompt: str) -> str:
    log.info("Starting LLM response...")
    for chunk in rag_chain_with_source(prompt):
        yield chunk


@app.get("/health")
def health_check():
    res = requests.get(f"http://{__LLM_HOST__}:{__LLM_PORT__}/health")
    return {"status": res.status_code}


@app.get("/generate")
def generate(prompt: str) -> str:
    system_instructions = "<s> [INST] You are a helpful AI assistant whose job is to answer questions as accurately as possible. [/INST] "
    prompt_template = PromptTemplate.from_template(
        "{system_instructions} User: {prompt} Assistant: "
    )
    prompt = prompt_template.format(
        system_instructions=system_instructions, prompt=prompt
    )

    resp = model.invoke(prompt)
    return resp


@app.get("/rag_generate")
def rag_generate(prompt: str) -> str:
    resp = retriever.prompt_builder(prompt)
    return resp


def start_server(host: str, port: int) -> None:
    log.info(f"Starting server: HOST: {host}, PORT: {port}...")
    uvicorn.run(app, host=host, port=port)


class Retriever:
    use_qdrant: bool = False
    chunk_size: int = 384
    """all-MiniLM-L6-v2 has a max token limit of 384."""
    chunk_overlap: int = 100
    """A value of 100 was chosen because of all-MiniLM-L6-v2's max token limit of 384."""
    vector_datastore: Union[Chroma, Qdrant] = None
    """Chroma and Qdrant are recommended, others can be added but haven't been tested."""
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_model: HuggingFaceEmbeddings = None
    """Use a local embedding model to avoid reaching out to the internet."""
    sim_thresh: float = 0.4
    embedding_filter: EmbeddingsFilter = None
    compression_retriever: ContextualCompressionRetriever = None
    documents: List[Document] = None
    """List of Document objects provided to the retriever as context."""
    retriever: VectorStoreRetriever = None
    model: Mixtral8x7b = Mixtral8x7b()

    def __init__(
        self,
        document_struct: Dict[str, Dict[str, Path]],
        embedding_model_name: str,
        model: BaseLLM,
    ):
        self.model = model
        self.embedding_model_name = embedding_model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.documents = self.load_documents(document_struct)
        self.documents = self.split_documents(self.documents)
        # The Qdrant variant is set to use :memory: as the location, which means it will
        # only be available in the current session. This is useful for testing but should
        # be changed for production.
        if self.use_qdrant:
            self.vector_datastore = Qdrant.from_documents(
                self.documents,
                self.embedding_model,
                location=":memory:",
                collection_name="acronyms",
            )
        else:
            self.vector_datastore = Chroma.from_documents(
                self.documents, self.embedding_model
            )
        self.retriever = self.vector_datastore.as_retriever()
        self.embedding_filter = EmbeddingsFilter(
            embeddings=self.embedding_model, similarity_threshold=self.sim_thresh
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.embedding_filter, base_retriever=self.retriever
        )

    def prompt_builder(self, prompt: str) -> str:
        header = "<s> [INST] You are a helpful AI assistant whose job is to answer questions as accurately as possible. [/INST] "
        prompt_template = PromptTemplate.from_template(
            "{header} User: {prompt} Assistant: "
        )
        compressed_docs = self.compression_retriever.get_relevant_documents(prompt)
        useful_information = []
        if len(compressed_docs) == 0:
            log.error("No documents found that match the threshold.")
            log.error("Current threshold: {self.sim_thresh}")
            log.error("Dispatching prompt without modifications!")
            prompt = prompt_template.format(header=header, prompt=prompt)
            return model.invoke(prompt)
        else:
            log.info("Found relavent documents")
            for doc in compressed_docs:
                log.info("====================================================")
                log.info(f"Document: {doc.page_content}")
                log.info("====================================================")
                splits = doc.page_content.split("\n")
                acronym, meaning = (
                    splits[0].split(":")[1].strip(),
                    splits[1].split(":")[1].strip(),
                )
                useful_information.append({"acronym": acronym, "meaning": meaning})
            for info in useful_information:
                header += f" [INST] This acronym is useful: {info['acronym']} = {info['meaning']} [/INST] "
            prompt = prompt_template.format(header=header, prompt=prompt)
            log.info(f"Dispatching prompt with modifications: {prompt}")
            return model.invoke(prompt)

    @staticmethod
    def pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
                [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
            )
        )

    def load_documents(self, documents: Dict[str, Dict[str, Path]]) -> List[Document]:
        retrieved_documents = []
        type_loader: Union[CSVLoader, JSONLoader] = None
        for doc_type in documents.keys():
            log.info(f"Loading storage type: {doc_type}...")
            match doc_type:
                case "csv":
                    type_loader = CSVLoader
                case "json":
                    type_loader = JSONLoader
            item_info = documents[doc_type]
            for item, path in item_info.items():
                log.info(f"Loading {item}...")
                if path.exists():
                    log.info(f"found it!")
                    log.info(f"Loading {item} data...")
                    csv_loader = type_loader(file_path=str(path))
                    data = csv_loader.load()
                    retrieved_documents += data
                    log.info(f"Loaded {item} data!")
                else:
                    log.critical(f"{item} not found!")
                    exit()
        return retrieved_documents

    def split_documents(self, data: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        documents = text_splitter.split_documents(data)
        return documents


if __name__ == "__main__":
    # region configuration
    # ===================================================================================

    # ===================================================================================
    # endregion

    # region Server setup
    # ===================================================================================
    # log.info("Starting server...")
    # log.info("Loading configuration files...")
    # config_path = Path("./config.toml").absolute()
    # if config_path.exists():
    #     conf = toml.load(str(config_path))
    # else:
    #     log.critical("config.toml not found!")
    #     exit()
    # ===================================================================================
    # endregion

    # region Load acronym data
    # ===================================================================================
    # log.info("Looking for acronym csv...")
    # acronym_path = Path("./acronyms.csv").absolute()
    # if acronym_path.exists():
    #     log.info("found it!")
    #     log.info("Loading acronym data...")
    #     acronym_path = str(acronym_path)
    #     csv_loader = CSVLoader(file_path=acronym_path)
    #     acronym_data = csv_loader.load()
    #     log.info("Loaded acronym data!")
    # else:
    #     log.critical("acronyms.csv not found!")
    #     exit()
    # ===================================================================================
    # endregion

    # region Vector store
    # ===================================================================================
    # log.info("Loading vectorstore...")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=100)
    # documents = text_splitter.split_documents(acronym_data)
    # ===================================================================================
    # endregion

    # Embedding model
    # ===================================================================================
    # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # qdrant = Qdrant.from_documents(
    #     documents,
    #     embedding_model,
    #     location=":memory:",  # Local mode with in-memory storage only
    #     collection_name="acronyms",
    # )
    # vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    # log.info("Loaded vectorstore!")
    # ===================================================================================

    # ===================================================================================
    log.info("Loading retriever...")
    documents = {
        "csv": {
            "acronyms": Path("./acronyms.csv"),
        }
    }
    model = Mixtral8x7b()
    retriever = Retriever(
        document_struct=documents, embedding_model_name="all-MiniLM-L6-v2", model=model
    )
    # retriever = qdrant.as_retriever()
    # prompt = hub.pull("rlm/rag-prompt")
    log.info("Loaded retriever!")

    # log.info("Loading model...")
    # log.info("Loaded model!")

    # log.info("Creating 'RAG' chain...")
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | model
    #     | StrOutputParser()
    # )
    # rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain)
    # log.info("Created 'RAG' chain!")

    start_server(__API_HOST__, int(__API_PORT__))
