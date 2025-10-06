import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import json

load_dotenv()


def get_client() -> Elasticsearch:
    url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    api_key = os.getenv("ELASTICSEARCH_API_KEY")
    api_key_id = os.getenv("ELASTICSEARCH_API_KEY_ID")

    if api_key_id and api_key:
        return Elasticsearch(url, api_key=(api_key_id, api_key))
    if api_key:
        return Elasticsearch(url, api_key=api_key)
    return Elasticsearch(url)


es = get_client()

# result = es.search(
#     index="patents",
#     body={
#         "query": {
#             "nested": {
#                 "path": "abstracts",
#                 "query": {
#                     "bool": {
#                         "must": [
#                             {
#                                 "match": {
#                                     "abstracts.text": {
#                                         "query": "piece goods",
#                                         "operator": "and",
#                                     }
#                                 }
#                             }
#                         ]
#                     }
#                 },
#                 "inner_hits": {
#                     "size": 5,
#                     "highlight": {"fields": {"abstracts.text": {}}},
#                 },
#             }
#         },
#         "size": 25,
#     },
# )
result = es.search(
    index="patents",
    body={
        "query": {
            "match": {
                "abstracts.text": {
                    "query": "display device",
                    "operator": "and"
                }
            }
        },
        "highlight": {
            "fields": {
                "abstracts.text": {},
            }
        },
        "size": 25,
    },
)

print(json.dumps(result["hits"], indent=2))

# es.delete_by_query(index="patents", body={"query": {"match_all": {}}})
