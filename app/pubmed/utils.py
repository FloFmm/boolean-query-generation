import re


def remove_tags(pubmed_query: str):
    pubmed_query = pubmed_query.replace("[mh]", "")
    pubmed_query = pubmed_query.replace("[mh:noexp]", "")
    pubmed_query = pubmed_query.replace("[tiab]", "")
    pubmed_query = pubmed_query.replace('"', "")
    return pubmed_query


def pubmed_query_to_lambda(expr: str):
    """
    Transform a boolean expression string into a Python lambda expression string.
    """
    # remove tags
    expr = remove_tags(expr)

    # get variables
    # --- Extract variable names (before replacement) ---
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_\[\]]*\b", expr)
    keywords = {"OR", "AND", "NOT"}
    variables = sorted(set(t for t in tokens if t not in keywords))

    # Normalize spacing
    expr = re.sub(r"\s+", " ", expr.strip())

    # Replace operators with Python equivalents
    expr = expr.replace("OR", "or")
    expr = expr.replace("AND", "and")

    # Handle NOT: "NOT ( ... )" -> "not ( ... )"
    expr = re.sub(r"\bNOT\s+", "not ", expr)

    # Insert implicit AND before NOT when needed:
    # "(A) NOT (B)" -> "(A) and not (B)"
    expr = re.sub(r"\s+not", " and not", expr)

    # Replace variables with dictionary lookups: d["var"]
    def repl_var(match):
        var = match.group(0)
        if var in {"or", "and", "not"}:
            return var
        return f'd["{var}"]'

    expr = re.sub(r"\b[a-zA-Z_][a-zA-Z0-9_\[\]]*\b", repl_var, expr)
    # Wrap into lambda
    return eval(f"lambda d: {expr}"), variables
