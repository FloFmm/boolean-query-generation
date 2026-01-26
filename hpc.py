queries = 300
runs = 1000
classifiers = 10
documents = 10_000_000
retrieval_sys = 5
print("CPU")
cpu_train = queries * runs * classifiers * retrieval_sys * 120 / 60 / 60
cpu_pubmed = queries * runs * classifiers * retrieval_sys * 0.1 / 60 / 60
cpu_corpus = documents * 120 / 60 / 60
cpu_total = cpu_train + cpu_corpus

print(
    f"train boolean classifier (e.g. decision trees) per query: "
    f"{queries}queries * {runs}runs * {classifiers}classifiers * {retrieval_sys}retrieval_sys * 120s/query / 60 / 60 = {cpu_train:.0f}h"
)

# print(f"querying pubmed api: max 10 request per second: "
#       f"{queries}queries * {runs}runs * {classifiers}classifiers * {retrieval_sys}retrieval_sys * 0.1s/query / 60 / 60 = {cpu_pubmed:.0f}h")

print(f"prepare corpus: {documents:,}docs * 120s/doc / 60 / 60 = {cpu_corpus:.0f}h")

print(f"→ CPU total = {cpu_total:.0f}h  + 20% buffer = ~1.000.000h")
print()

print("GPU")
gpu_embed = documents * 0.073 / 60 / 60
gpu_llm = queries * runs * classifiers * retrieval_sys * 1.96 / 60 / 60
# gpu_semantic = queries * runs * classifiers * 2.5 / 60 / 60
gpu_total = gpu_embed + gpu_llm  # + gpu_semantic

print(f"corpus embedding: {documents:,}docs * 0.073s / 60 / 60 = {gpu_embed:.0f}h")

print(
    f"LLM boolean query generation/expansion: "
    f"{queries}queries * {runs}runs * {classifiers}classifiers * {retrieval_sys}retrieval_sys * 1.96s / 60 / 60 = {gpu_llm:.0f}h"
)

# print(f"semantic and LLM driven query expansion: "
#       f"{queries}queries * {runs}runs * {classifiers}classifiers * 2.5s / 60 / 60 = {gpu_semantic:.0f}h")

print(f"→ GPU total = {gpu_total:.0f}h + 20% buffer = ~10.000h")
