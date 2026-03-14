import re

from app.config.config import COLORS


def math_replace(text):
    # relace any occurence of 2 or more letters in a row starting with " and ending with " by \mathrm{...}
    text = re.sub(r'"([a-zA-Z 1-9\-\#]{2,})"', r"\\mathrm{\1}", text)
    text = text.replace("$", "")
    text = text.replace('("', "\\mathrm{")
    text = text.replace('"', "")
    text = text.replace(")", "}")
    text = text.replace("(", "{")
    text = text.replace("#sym.", "\\")
    text = text.replace(" _", "_")
    text = text.replace("_ ", "_")
    text = text.replace(" dot ", " \\cdot ")
    text = re.sub(r"(?<!\\) ", r"\\ ", text)
    text = "$" + text + "$"
    return text


def generate_replacements():
    assets_path = "../master-thesis-writing/writing/thesis/assets/assets.typ"

    with open(assets_path) as f:
        content = f.read()

    replacements = {}
    for m in re.finditer(r'^#let ([\w-]+) = (.+["|\]|$])', content, re.MULTILINE):
        name = m.group(1)
        desc = m.group(2).strip().strip(" ").strip('"').strip("[]")
        replacements[name] = desc

    keep_running = True
    while keep_running:
        keep_running = False
        for k1, v1 in replacements.items():
            for k2, v2 in replacements.items():
                if "#" + k2 in v1:
                    new = v1.replace("#" + k2, v2)
                    if new[0] == "$" and new[-1] == "$":
                        new = math_replace(new)
                    replacements[k1] = new
                    keep_running = True

    final = {}
    for k, desc in replacements.items():
        if desc[0] == "$" and desc[-1] == "$":
            desc = math_replace(desc)
        desc = desc.replace("*OR*", "OR")
        desc = desc.replace("#OR", "OR")
        final[k] = desc
    return final


REPLACEMENTS = generate_replacements()


def pretty_print_param(name: str, break_long=False) -> str:
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
        # "fixed_k": r"fixed-$k$",
        # "cosine_k": r"cosine-$k$",
        # "pos_count_k": r"pos-count-$k$",
        # "top_k": r"$k$",
        "randomize": "Rand.",
        "Randomize": "Rand.",
        "impurity": "Imp.",
        "Impurity": "Imp.",
        "decrease": "Decr.",
        "Decrease": "Decr.",
        "(leaf nodes)": "(Leaf)",
        "(Leaf Nodes)": "(Leaf)",
        "(root node)": "(Root)",
        "(Root Node)": "(Root)",
    }
    if name in REPLACEMENTS and name + "_long" in REPLACEMENTS:
        name = f"{REPLACEMENTS[name + '_long']} ({REPLACEMENTS[name]})"
    else:
        for old, new in sorted(
            REPLACEMENTS.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            if "#" + old in name:
                print("name0", name)
                name = name.replace("#" + old, new)
                print("name1", name)
                if name[0] == "$" and name[-1] == "$":
                    name = math_replace(name)
                print("name2", name)
                break

    # Apply generic cosmetic replacements last.
    for old, new in sorted(
        replacements.items(), key=lambda kv: len(kv[0]), reverse=True
    ):
        name = name.replace(old, new)
    if break_long and name.count("(") > 0 and len(name) > 20:
        # break last (
        last_idx = name.rfind("(")
        name = (
            name[:last_idx] + "\n" + name[last_idx:].replace("(", "").replace(")", "")
        )
        
    print("last name", name)
    return name


def prettify_axes(ax, break_long=False):
    """
    Apply pretty_print_param to all text labels in an axes object.
    Handles xlabel, ylabel, title, and tick labels.
    """
    # Pretty print axis labels
    if ax.get_xlabel():
        ax.set_xlabel(pretty_print_param(ax.get_xlabel(), break_long=break_long))
    if ax.get_ylabel():
        ax.set_ylabel(pretty_print_param(ax.get_ylabel()))
    if ax.get_title():
        ax.set_title(pretty_print_param(ax.get_title()))

    # Pretty print tick labels
    xticklabels = [
        pretty_print_param(label.get_text()) for label in ax.get_xticklabels()
    ]
    if any(xticklabels):
        ax.set_xticklabels(xticklabels)

    yticklabels = [
        pretty_print_param(label.get_text()) for label in ax.get_yticklabels()
    ]
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
        text = text.replace(
            f"\) {o} \(", f"#text(size: {text_size}, [*\)* *{o}* *\(*])"
        )  # #emph(text(3pt)[(])
        # text = text.replace(f" {o} ((", f"#text(size: 1.1em, [ *{o}* *(*])//") # #emph(text(3pt)[(])
        text = text.replace(
            f" {o} \(", f"#text(size: {text_size}, [ *{o}* *\(*])"
        )  # #emph(text(3pt)[(])
        text = text.replace(
            f"\) {o} ", f"#text(size: {text_size}, [*\)* *{o}* ])"
        )  # #emph(text(3pt)[(])

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
    return [
        t.strip()
        for t in re.split(r"\s+AND\s+|\s+OR\s+|\s+NOT\s+|\(|\)", query)
        if t.strip()
    ]


def highlight_query_words(
    query_text: str,
    words: set,
    color: str,
    fmt: str = "highlight",
    lightness: float = 30.0,
) -> str:
    # query_text = escape_typst(query)
    # if not words:
    #     return query_text

    for term in sorted(words, key=len, reverse=True):
        escaped_term = escape_typst(term)
        regex_term = re.escape(escaped_term)
        if fmt == "underline":
            replacement = f"#underline(stroke: 1pt + {color})[{escaped_term}]"
        else:
            replacement = (
                f"#highlight(fill: {color}.lighten({lightness}%))[{escaped_term}]"
            )

        # Match term bounded by operators / parens / string boundaries on both sides
        pattern = (
            r"((?:^|\[|\\\(|\\\[|\\#| (?:OR|AND|NOT) ))"
            + regex_term
            + r"(?= (?:OR|AND|NOT) |\\\)|\\\]|\]|\\#|$)"
        )
        old_query_text = query_text
        query_text = re.sub(pattern, lambda m: m.group(1) + replacement, query_text)
        assert old_query_text != query_text, (
            f"Failed to replace term '{term}' in query text: {query_text}"
        )

    return query_text


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def value_to_marking(
    w,
    value,
    minimum_of_all_replacements,
    maximum_of_all_replacements,
    min_alpha=0.0,
    min_value_pct=0.01,
):
    if value < 0:
        if abs(value) < min_value_pct * abs(minimum_of_all_replacements):
            return None
        color = COLORS["positive"]
        lighting_factor = min_alpha + abs(value / minimum_of_all_replacements) * (
            1 - min_alpha
        )
        # lighten color by vlaue
        color = mcolors.to_rgba(color)
        color = (color[0], color[1], color[2], lighting_factor)
        color = mcolors.to_hex(color, keep_alpha=True)
        return (w, "highlight", f'rgb("{color}")')
    else:
        if abs(value) < min_value_pct * abs(maximum_of_all_replacements):
            return None
        color = COLORS["negative"]
        lighting_factor = min_alpha + abs(value / maximum_of_all_replacements) * (
            1 - min_alpha
        )
        # lighten color by vlaue
        color = mcolors.to_rgba(color)
        color = (color[0], color[1], color[2], lighting_factor)
        color = mcolors.to_hex(color, keep_alpha=True)
        return (w, "underline", f'rgb("{color}")')


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
        assert old_query_text != query, (
            f"Failed to replace term '{old_word}' in query text: {query}"
        )
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
