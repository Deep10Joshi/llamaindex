# pip install llama_index llama_index.embeddings.huggingface llama_index.llms.huggingface chromadb llama-index-vector-stores-chroma

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI, HuggingFaceLLM, TextGenerationInference
from llama_index.core import PromptTemplate
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
import json


class LLMBaseModel:
    def __init__(self):
        self.LLM_MODEL = HuggingFaceInferenceAPI(
            model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
            # model_name="microsoft/Phi-3-mini-4k-instruct",
            # model_name="google/gemma-1.1-2b-it",
            token="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",
        )

        self.EMBED_MODEL = HuggingFaceEmbedding(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            text_instruction="Given are the offers we provide, where each offer is uniqely identified by its offering_id",
            query_instruction="Retrieve all the relevent offering_ids from the given query"
        )


        self.PERSIST_DIR = "./vector-indexes/" + str(self.EMBED_MODEL.model_name)
        self.DATA_DIR = "./"
        self.VECTOR_INDEX = None


        self.CHROMA_DB = chromadb.Client()
        self.CHROMA_COLLECTION = self.CHROMA_DB.get_or_create_collection("Categories")


        Settings.embed_model = self.EMBED_MODEL
        Settings.llm = self.LLM_MODEL

    # Custom printer
    def notifyMessage(self, text):
        print(f"{'='*20}{text}{'='*20}")

    # Builds nodes from the documents
    def createDocs(self):
        self.docs = SimpleDirectoryReader(self.DATA_DIR).load_data(show_progress=True)
        parser = SentenceSplitter(paragraph_separator="},\n")
        self.nodes = parser.get_nodes_from_documents(self.docs, show_progress=True)
        self.notifyMessage(f"Parsed {len(self.nodes)} Nodes")

    # Saves the index into persist directory
    def saveVectorIndexToDisk(self):
        self.VECTOR_INDEX.storage_context.persist(persist_dir=self.PERSIST_DIR)
        self.notifyMessage("Vector Index Saved")



# ====================Chroma DB======================
    def createIndexFromChromaStorage(self):
        self.createDocs()

        # Creating chroma vector store
        chroma_vector_store = ChromaVectorStore(self.CHROMA_COLLECTION, persist_dir=self.PERSIST_DIR)

        # creating a storage context out of the chroma vector store
        chroma_storage_context = StorageContext.from_defaults(vector_store=chroma_vector_store)

        # Building the index by using the storage context
        self.VECTOR_INDEX = VectorStoreIndex(
            nodes=self.nodes,
            storage_context=chroma_storage_context,
            show_progress=True,
            embed_model=self.EMBED_MODEL
        )

        # Saving the index into the disk
        self.saveVectorIndexToDisk()

    def loadIndexFromChromaStorage(self):
        # Creating chroma vector store
        chroma_vector_store = ChromaVectorStore(self.CHROMA_COLLECTION, persist_dir=self.PERSIST_DIR)

        # Building the index from the loaded vector store
        self.VECTOR_INDEX = VectorStoreIndex.from_vector_store(
            chroma_vector_store,
            embed_model=self.EMBED_MODEL
        )
        self.notifyMessage("Chroma Vector Index Loaded")


# ==================== Default DB ========================
    def createIndexFromDefaultStorage(self):
        self.createDocs()
        self.VECTOR_INDEX = VectorStoreIndex(
            nodes=self.nodes,
            show_progress=True
        )

        # Saving the index into the disk
        self.saveVectorIndexToDisk()


    def loadIndexFromDefaultStorage(self):
        # creating a storage context from defaults
        storage_context = StorageContext.from_defaults(
            persist_dir=self.PERSIST_DIR,
        )

        # Building the index from the storage context
        self.VECTOR_INDEX = load_index_from_storage(
            storage_context=storage_context
        )
        self.notifyMessage("Vector Index Loaded")


    def generateResponse(self, userPrompt:str):
        template = (
            "We have provided context information below. \n"
            "---------------------\n"
            "You are a JSON search engine whose role is to find all the offering id's of offers where the details or the tag matches content of the question being asked"
            "\n---------------------\n"
            "You should not provide any extra details, only retrieve the offering_id"
            "\n---------------------\n"
            "The output should be in this format: '[offering id's]' "
            "\n---------------------\n"
            "Given this information, please answer the question: give various offer details for {query_str}\n"
        )
        qa_template = PromptTemplate(template)
        message = qa_template.format_messages(query_str=userPrompt)[0].content
        # print(message)

        # print(self.VECTOR_INDEX.as_retriever(similarity_top_k=2).retrieve(userPrompt))
        self.QUERY_ENGINE = self.VECTOR_INDEX.as_query_engine(
            similarity_top_k=5,
        )
        response = self.QUERY_ENGINE.query(message)
        # print(response)
        return response