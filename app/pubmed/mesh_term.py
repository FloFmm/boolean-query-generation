import urllib.request
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import json

def strip_mesh_term(term: str) -> str:
    term = term.strip()

    if term.startswith("(") and term.endswith(")"):
        term = term[1:-1].strip()

    return term.lower()

def get_ancestors_by_name(mesh_data, descriptor_name):
    """
    Given a descriptor name, return ALL ancestor names in the MeSH hierarchy.
    """

    # name -> list of UIs
    name_to_ui = {
        data["name"].lower(): ui for ui, data in mesh_data.items()
    }

    if descriptor_name not in name_to_ui:
        print(f"warning {descriptor_name} not in mesh term data")
        return {descriptor_name}
        # raise ValueError(f"Descriptor not found: {descriptor_name}")

    ui = name_to_ui[descriptor_name]
    record = mesh_data[ui]

    ancestors = set()
    tree_to_name = {}

    # Reverse index: tree number -> name
    for item in mesh_data.values():
        for tn in item["tree_numbers"]:
            tree_to_name[tn] = item["name"]

    for tn in record["tree_numbers"]:
        parts = tn.split(".")
        # progressively shorten the tree number
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix in tree_to_name:
                ancestors.add(tree_to_name[prefix].lower())

    return list(ancestors)

def _parse_mesh_xml(xml_path: Path):
    print(f"[OK] Parsing XML → {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    mesh_data = {}

    for desc in root.findall(".//DescriptorRecord"):
        ui = desc.findtext("DescriptorUI")
        name = desc.findtext("DescriptorName/String")
        scope_note = desc.findtext("ScopeNote")

        tree_numbers = [
            tn.text for tn in desc.findall("./TreeNumberList/TreeNumber")
        ]

        concepts = []
        for concept in desc.findall("./ConceptList/Concept"):
            c_name = concept.findtext("ConceptName/String")
            terms = [
                t.findtext("String") for t in concept.findall("./TermList/Term")
            ]
            concepts.append({"name": c_name, "terms": terms})

        mesh_data[ui] = {
            "ui": ui,
            "name": name,
            "tree_numbers": tree_numbers,
            "concepts": concepts,
            "scope_note": scope_note,
        }

    print(f"[OK] Parsed {len(mesh_data)} descriptors")
    return mesh_data

def _save_mesh_json(mesh_data, json_path: Path):
    print(f"[OK] Saving JSON → {json_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mesh_data, f, ensure_ascii=False, indent=2)

def download_mesh_xml(year=2025, target_dir="data/pubmed/mesh_data"):
    """
    Ensures MeSH data is available in JSON form.
    
    Workflow:
      1. If JSON exists → load & return it.
      2. Else if XML exists → parse it and save JSON.
      3. Else → download ZIP → extract → parse XML → save JSON.
    
    Returns:
        mesh_data (dict keyed by UI)
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    xml_path = target_dir / f"desc{year}.xml"
    zip_path = target_dir / f"desc{year}.zip"
    json_path = target_dir / f"desc{year}.json"

    # ---------------------------------------------------------------
    # STEP 1 — JSON already exists → load and return
    # ---------------------------------------------------------------
    if json_path.exists():
        print(f"[OK] Loading cached MeSH JSON → {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------------------------------------------------------------
    # STEP 2 — XML exists but JSON does not → parse XML
    # ---------------------------------------------------------------
    if xml_path.exists():
        print(f"[OK] Using existing XML → {xml_path}")
        mesh_data = _parse_mesh_xml(xml_path)
        _save_mesh_json(mesh_data, json_path)
        return mesh_data

    # ---------------------------------------------------------------
    # STEP 3 — XML missing → download and extract ZIP
    # ---------------------------------------------------------------
    url = f"https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{year}.zip"
    print(f"[DOWNLOAD] {url}")

    urllib.request.urlretrieve(url, zip_path)
    print(f"[OK] Downloaded ZIP → {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
        print(f"[OK] Extracted ZIP into → {target_dir}")

    if not xml_path.exists():
        raise FileNotFoundError(f"{xml_path} not found after extracting ZIP!")

    # ---------------------------------------------------------------
    # Parse XML → save JSON
    # ---------------------------------------------------------------
    mesh_data = _parse_mesh_xml(xml_path)
    _save_mesh_json(mesh_data, json_path)
    return mesh_data

def expand_mesh_terms(mesh_list, mesh_ancestor_data=None):
    """
    Expands compact MeSH-style notations like
    'Down Syndrome/blood/*prevention & control'
    into full term combinations.
    """
    expanded = set()
    for mesh_str in mesh_list:
        mesh_str = mesh_str.replace('&', 'and')
        mesh_str = mesh_str.replace('*', '')
        parts = mesh_str.split('/')
        striped_mesh = parts[0]
        expanded.add(striped_mesh)
        if mesh_ancestor_data:
            expanded.update(get_ancestors_by_name(mesh_ancestor_data, striped_mesh))
        for p in parts[1:]:
            expanded.add(f"{parts[0]}/{p}")

    return expanded


if __name__ == "__main__":
    mesh = download_mesh_xml(2025)
    print("Loaded", len(mesh), "descriptors")
    anc = get_ancestors_by_name(mesh, "Humans")
    print("Ancestors:", anc)