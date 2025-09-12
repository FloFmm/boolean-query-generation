from pydantic import BaseModel
from typing import List, Dict, Optional


class Node(BaseModel):
    id: str
    title: str
    abstract: Optional[str]
    pos: List[float]


class Edge(BaseModel):
    source: str
    target: str
    weight: float


class SearchResponse(BaseModel):
    query_node: Node
    nodes: List[Node]
    edges: List[Edge]


class TrainRequest(BaseModel):
    relevant_ids: List[str]
    non_relevant_ids: List[str]


class TrainResponse(BaseModel):
    tree_text: str
    formula: str
