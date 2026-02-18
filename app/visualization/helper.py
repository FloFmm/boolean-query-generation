import re


def pretty_print_param(name: str) -> str:
    """
    Transform parameter names into pretty-printed versions.
    E.g., 'max_depth' -> 'max depth', 'n_estimators' -> 'n estimators'
    """
    mapping = {
        "min_impurity_decrease_range_end": "min imp. decr. leaf",
        "min_impurity_decrease_range_start": "min imp. decr. root",
        "randomize_min_impurity_decrease_range": "randomize min imp. decr.",
        "min_weight_fraction_leaf": "min leaf weight",
        "term_expansions": "word exp",
        "mh_noexp": "MeSH no exp",
        "top_k_or_candidates": "k OR candidates",
        "paths": "Rules",
        "added_ORs": "Inner ORs",
        "avg_path_len": "Avg. Rule Length",
        "avg_term_len": "Avg. Disj. Length",
        "ops_count": "Logical Operators",
        "synonym_ORs": "Synonym ORs",
        "all_ORs": "ORs",
        "avg_df": "Median DF",
        "duplicate_pct_exact": "Exact Duplicates %",
        "duplicate_pct_substring": "Substr. Duplicates %",
    }
    replacements = {
        "randomize": "rand.",
        "impurity": "imp.",
        "decrease": "decr.",
    }
    name = mapping.get(name, name)
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("_", " ")


def prettify_axes(ax):
    """
    Apply pretty_print_param to all text labels in an axes object.
    Handles xlabel, ylabel, title, and tick labels.
    """
    # Pretty print axis labels
    if ax.get_xlabel():
        ax.set_xlabel(pretty_print_param(ax.get_xlabel()))
    if ax.get_ylabel():
        ax.set_ylabel(pretty_print_param(ax.get_ylabel()))
    if ax.get_title():
        ax.set_title(pretty_print_param(ax.get_title()))
    
    # Pretty print tick labels
    xticklabels = [pretty_print_param(label.get_text()) for label in ax.get_xticklabels()]
    if any(xticklabels):
        ax.set_xticklabels(xticklabels)
    
    yticklabels = [pretty_print_param(label.get_text()) for label in ax.get_yticklabels()]
    if any(yticklabels):
        ax.set_yticklabels(yticklabels)
        

def escape_typst(text: str) -> str:
    """Escape special characters for Typst."""
    if text is None:
        return ""

    # Escape backslash first to avoid double-escaping
    text = str(text).replace("\\", "\\\\")

    # Escape other special characters as needed
    # In Typst, special characters in content might need escaping
    # Common ones: [ ] # { } < >
    text = text.replace("[", "\\[")
    text = text.replace("]", "\\]")
    text = text.replace("#", "\\#")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("*", "\\*")
    text = text.replace(")", "\\)")
    text = text.replace("(", "\\(")
    
    text = strip_matching_outer_parens(text)
    return text

def mark_outer_operators(query, operator_types):
    import re

    text = query
    text_size = "1.1em"
    for o in operator_types:
        # text = text.replace(f") {o} ((", f"#text(size: 1.1em, [*)* *{o}* *(*])//") # #emph(text(3pt)[(])
        text = text.replace(f"\) {o} \(", f"#text(size: {text_size}, [*\)* *{o}* *\(*])") # #emph(text(3pt)[(])
        # text = text.replace(f" {o} ((", f"#text(size: 1.1em, [ *{o}* *(*])//") # #emph(text(3pt)[(])
        text = text.replace(f" {o} \(", f"#text(size: {text_size}, [ *{o}* *\(*])") # #emph(text(3pt)[(])
        text = text.replace(f"\) {o} ", f"#text(size: {text_size}, [*\)* *{o}* ])") # #emph(text(3pt)[(])

    # Mark a leading "\(" even if it starts after a line break ("\\ ")
    if text.endswith("\)"):
        text = text[:-2] + f"#text(size: {text_size}, [*\)*])"
    if text.startswith("\("):
        text = f"#text(size: {text_size}, [*\(*])" + text[2:]
    return text


def strip_matching_outer_parens(text: str) -> str:
    """Remove matching outer escaped parentheses if they wrap the entire string."""
    if not (text.startswith("\\(") and text.endswith("\\)")):
        return text

    depth = 0
    i = 0
    closing_idx = None
    while i < len(text):
        if text.startswith("\\(", i):
            depth += 1
            i += 2
            continue
        if text.startswith("\\)", i):
            depth -= 1
            i += 2
            if depth == 0:
                closing_idx = i - 2
                break
            continue
        i += 1

    if closing_idx == len(text) - 2:
        return text[2:-2]
    return text

def split_query_into_words(query):
    return [t.strip() for t in re.split(r'\s+AND\s+|\s+OR\s+|\s+NOT\s+|\(|\)', query) if t.strip()]

def highlight_query_words(query_text: str, words: set, color: str, fmt: str = "highlight", lightness: float = 30.0) -> str:
    # query_text = escape_typst(query)
    # if not words:
    #     return query_text

    for term in sorted(words, key=len, reverse=True):
        escaped_term = escape_typst(term)
        regex_term = re.escape(escaped_term)
        if fmt == "underline":
            replacement = f"#underline(stroke: 1pt + {color})[{escaped_term}]"
        else:
            replacement = f"#highlight(fill: {color}.lighten({lightness}%))[{escaped_term}]"

        # Match term bounded by operators / parens / string boundaries on both sides
        pattern = (
            r"((?:^|\\\(|\\\[|\\#| (?:OR|AND|NOT) ))"
            + regex_term
            + r"(?= (?:OR|AND|NOT) |\\\)|\\\]|\\#|$)"
        )
        query_text = re.sub(pattern, lambda m: m.group(1) + replacement, query_text)

    return query_text
       