import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def custom_split_by_headers(documents, separator="###"):
    split_docs = []
    for doc in documents:
        parts = doc.page_content.split(separator)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                split_docs.append(Document(page_content=cleaned, metadata=doc.metadata))
    return split_docs

def generate_faiss_indexes(
    root_docs_dir="documentos",
    output_dir="faiss_indexes",
    embedding_model_name="intfloat/multilingual-e5-small",
    separator="###"
):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    os.makedirs(output_dir, exist_ok=True)

    for collection_name in os.listdir(root_docs_dir):
        collection_path = os.path.join(root_docs_dir, collection_name)
        if not os.path.isdir(collection_path):
            continue

        print(f"Procesando colección: {collection_name}")

        documents = []
        for filename in os.listdir(collection_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(collection_path, filename)
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                documents.extend(docs)

        docs_split = custom_split_by_headers(documents, separator=separator)
        print(f"Generando embeddings y creando FAISS para {collection_name} con {len(docs_split)} fragmentos...")

        faiss_index = FAISS.from_documents(docs_split, embeddings)

        collection_output_path = os.path.join(output_dir, collection_name)
        os.makedirs(collection_output_path, exist_ok=True)

        faiss_index.save_local(collection_output_path)

        print(f"FAISS guardado en {collection_output_path}")

if __name__ == "__main__":
    generate_faiss_indexes()
