from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict

class Retriever:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
        self.indexes: Dict[str, FAISS] = {}

        self._load_indexes()

    def _load_indexes(self):
        import os
        for name in os.listdir(self.base_path):
            path = os.path.join(self.base_path, name)
            if os.path.isdir(path):
                index = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
                self.indexes[name] = index
                print(f"Índice cargado: {name} con {len(index.index_to_docstore_id)} docs")

    def search(self, collection_id: str, query: str, k: int = 5):
        if collection_id not in self.indexes:
            raise ValueError(f"Colección no encontrada: {collection_id}")
        return self.indexes[collection_id].similarity_search(query, k=k)
