import re

from app.config.config import COLORS


def generate_replacements():
    assets_path = "../master-thesis-writing/writing/thesis/assets/assets.typ"

    with open(assets_path) as f:
        content = f.read()

    # Parse: #let <name> = $<formula>$  (inline math, no nested $)
    shorts = {}
    # for m in re.finditer(r'#let (\w+) = \$(.+?)\$', content):
    #     name, formula = m.group(1), m.group(2).strip()
    #     if not name.endswith("_long"):
    #         shorts[name] = formula

    # Parse: #let <name>_long = <anything on the line>
    longs = {}
    for m in re.finditer(r'^#let (\w+)_long = (.+)$', content, re.MULTILINE):
        name = m.group(1)
        desc = m.group(2).strip().strip(' ').strip('"').strip('[]')
        longs[name] = desc

    replacements = {}
    for name, long_val in longs.items():
        if name in shorts:
            replacements[name] = f"{long_val} (${shorts[name]}$)"
        else:
            replacements[name] = long_val

    return replacements


REPLACEMENTS = generate_replacements()

def pretty_print_param(name: str) -> str:
    """
    Transform parameter names into pretty-printed versions.
    E.g., 'max_depth' -> 'max depth', 'n_estimators' -> 'n estimators'
    """
    replacements = {
        # "min_impurity_decrease_range_end": "min imp. decr. leaf",
        # "min_impurity_decrease_range_start": "min imp. decr. root",
        # "randomize_min_impurity_decrease_range": "randomize min imp. decr.",
        # "min_weight_fraction_leaf": "min leaf weight",
        # "term_expansions": "word exp",
        # "mh_noexp": "MeSH no exp",
        # "top_k_or_candidates": "k OR candidates",
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
        "Objective Value": r"Objective Value $F_{50}$",
        "#OR": "OR",
        # "fixed_k": r"fixed-$k$",
        # "cosine_k": r"cosine-$k$",
        # "pos_count_k": r"pos-count-$k$",
        # "top_k": r"$k$",
        "randomize": "rand.",
        "impurity": "imp.",
        "decrease": "decr.",
        "(leaf": "\n(leaf",
        "(root": "\n(root",
    }
    # If the label is an exact parameter key, use the long form directly.
    if name in REPLACEMENTS:
        name = REPLACEMENTS[name]

    # Apply key-based replacements longest-first to avoid prefix collisions
    # like min_impurity_decrease before min_impurity_decrease_range_end.
    for old, new in sorted(REPLACEMENTS.items(), key=lambda kv: len(kv[0]), reverse=True):
        name = name.replace(old, new)

    # Apply generic cosmetic replacements last.
    for old, new in sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True):
        name = name.replace(old, new)

    return name.title()


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
            r"((?:^|\[|\\\(|\\\[|\\#| (?:OR|AND|NOT) ))"
            + regex_term
            + r"(?= (?:OR|AND|NOT) |\\\)|\\\]|\]|\\#|$)"
        )
        old_query_text = query_text
        query_text = re.sub(pattern, lambda m: m.group(1) + replacement, query_text)
        assert old_query_text != query_text, f"Failed to replace term '{term}' in query text: {query_text}"

    return query_text
       
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def value_to_marking(w, value, minimum_of_all_replacements, maximum_of_all_replacements, min_alpha=0.0, min_value_pct=0.01):
    if value < 0:
        if abs(value) < min_value_pct * abs(minimum_of_all_replacements):
            return None
        color = COLORS["positive"]
        lighting_factor = min_alpha + abs(value/minimum_of_all_replacements) * (1 - min_alpha)
        # lighten color by vlaue
        color = mcolors.to_rgba(color)
        color = (color[0], color[1], color[2], lighting_factor)
        color = mcolors.to_hex(color, keep_alpha=True)
        return (w, "highlight", f"rgb(\"{color}\")")
    else:
        if abs(value) < min_value_pct * abs(maximum_of_all_replacements):
            return None
        color = COLORS["negative"]
        lighting_factor = min_alpha + abs(value/maximum_of_all_replacements) * (1 - min_alpha)
        # lighten color by vlaue
        color = mcolors.to_rgba(color)
        color = (color[0], color[1], color[2], lighting_factor)
        color = mcolors.to_hex(color, keep_alpha=True)
        return (w, "underline", f"rgb(\"{color}\")")

def replace_word_in_query(query, old_word, new_word, replace_all=True):
    
    escaped = re.escape(old_word)
    pattern = (
        r"((?:^|\[|\\\(|\(|\\\[|\\#| (?:OR|AND|NOT) ))"
        + escaped
        + r"(?= (?:OR|AND|NOT) |\\\)|\)|\\\]|\]|\\#|$)"
    )
    old_query_text = query
    if replace_all:
        if old_word == new_word:
            return query
        query = re.sub(pattern, lambda m: m.group(1) + new_word, query)
        assert old_query_text != query, f"Failed to replace term '{old_word}' in query text: {query}"
        return query
    else:
        if old_word == new_word:
            return [query]
        # replace each occurrence individually and return array of modified queries
        queries = []
        for match in re.finditer(pattern, query):
            # match.group(1) is the prefix, match.span() is the full match
            start, end = match.span()
            prefix = match.group(1)
            # The matched word is right after the prefix, so replace only that
            word_start = match.start(1) + len(prefix)
            word_end = end
            new_query = query[:word_start] + new_word + query[word_end:]
            queries.append(new_query)
        return queries