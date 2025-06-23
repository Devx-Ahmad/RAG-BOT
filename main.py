from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from app.chatbot import handle_query, initialize_embeddings
import uvicorn
import os
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    initialize_embeddings()
    yield
    

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    return {"reply": handle_query(req.query)}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=8010, reload=True)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
