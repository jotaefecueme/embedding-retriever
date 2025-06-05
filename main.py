import logging
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI()
retriever = Retriever(base_path="./faiss_indexes")

class QueryRequest(BaseModel):
    id: str        
    question: str  
    k: int = 5     

@app.post("/query")
async def query_embeddings(req: QueryRequest):
    start_time = time.perf_counter()
    logger.info(f"RETRIEVER - Consulta recibida - id: {req.id}, pregunta: {req.question}, k: {req.k}")
    try:
        results = retriever.search(req.id, req.question, req.k)
        duration = (time.perf_counter() - start_time) * 1000  # ms
        logger.info(f"RETRIEVER - Consulta procesada en {duration:.2f} ms, resultados: {len(results)}")
        response = [
            {"content": doc.page_content}
            for doc in results
        ]
        return {"results": response}
    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
