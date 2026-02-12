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
        