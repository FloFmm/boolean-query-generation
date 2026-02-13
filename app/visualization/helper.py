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

    return text
        