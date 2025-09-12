from app.backend.schemas import SearchResponse
from app.backend.data.pubmed.search import search_pubmed_med_cpt  # your function


def run_search(query: str, top_k: int = 10) -> SearchResponse:
    result = search_pubmed_med_cpt(query, top_k=top_k)
    return SearchResponse(**result)
