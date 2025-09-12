from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.backend.schemas import SearchResponse, TrainRequest, TrainResponse
from app.backend.search import run_search
from app.backend.ml import text_tree_model

app = FastAPI()

# allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/search", response_model=SearchResponse)
def search_endpoint(query: str, top_k: int = 20):
    return run_search(query, top_k=top_k)


@app.post("/train_tree", response_model=TrainResponse)
def train_tree(req: TrainRequest):
    """
    Expects relevant and non-relevant texts (or pmids mapped to texts)
    """
    set1 = req.relevant_texts
    set2 = req.non_relevant_texts
    result = text_tree_model.train(set1, set2)

    return TrainResponse(
        tree_text=result["decision_tree"],
        formula=result["boolean_function_set1"],
    )
