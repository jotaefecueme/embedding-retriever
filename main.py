from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import Retriever

app = FastAPI()

retriever = Retriever(base_path="./faiss_indexes")

class QueryRequest(BaseModel):
    id: str        
    question: str  
    k: int = 5     

@app.post("/query")
async def query_embeddings(req: QueryRequest):
    try:
        results = retriever.search(req.id, req.question, req.k)
        response = [
            {
                "content": doc.page_content
            }
            for doc in results
        ]
        return {"results": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
