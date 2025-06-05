from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Retriever:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.embedding_model_name = "intfloat/multilingual-e5-base"
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        self.faiss_indices = {}

    
    def load_index(self, collection_id: str):
        if collection_id in self.faiss_indices:
            return  # Ya cargado
        index_path = f"{self.base_path}/{collection_id}"
        self.faiss_indices[collection_id] = FAISS.load_local(
            index_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
    )


    def search(self, collection_id: str, query: str, k: int = 5):
        if collection_id not in self.faiss_indices:
            self.load_index(collection_id)
        results = self.faiss_indices[collection_id].similarity_search(query, k=k)
        return results
