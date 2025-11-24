def expand_mesh_terms(mesh_list):
    """
    Expands compact MeSH-style notations like
    'Down Syndrome/blood/*prevention & control'
    into full term combinations.
    """
    expanded = {}
    for mesh_str in mesh_list:
        mesh_str = mesh_str.replace('&', 'and')
        mesh_str = mesh_str.replace('*', '')
        parts = mesh_str.split('/')
        expanded.add(parts[0])
        for p in parts[1:]:
            expanded.add(f"{parts[0]}/{p}")

    return expanded