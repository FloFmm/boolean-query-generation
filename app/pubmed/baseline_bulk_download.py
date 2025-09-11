import os
import gzip
import json
import xml.etree.ElementTree as ET
import shutil
from ftplib import FTP

# FTP server details
FTP_HOST = 'ftp.ncbi.nlm.nih.gov'
FTP_DIR = '/pubmed/baseline/'

# Local directory to save files
LOCAL_DIR = './data/pubmed/baseline'

os.makedirs(LOCAL_DIR, exist_ok=True)

# Connect to FTP server
ftp = FTP(FTP_HOST)
ftp.login()
ftp.cwd(FTP_DIR)

# List XML files
files = ftp.nlst()
xml_files = [f for f in files if f.endswith('.xml.gz')]

for xml_file in xml_files:
    local_gz_path = os.path.join(LOCAL_DIR, xml_file)
    local_xml_path = os.path.join(LOCAL_DIR, xml_file.replace('.gz', ''))
    local_jsonl_path = os.path.join(LOCAL_DIR, xml_file.replace('xml.gz', '.jsonl'))

    # Skip download if JSONL already exists
    if os.path.exists(local_jsonl_path) and os.path.exists(local_gz_path):
        print(f"Skipping {xml_file}, already converted to JSONL")
        continue
    
    attempt = 1
    while attempt < 3:
        try:
            # downlaod .gz file
            with open(local_gz_path, 'wb') as f:
                ftp.retrbinary(f'RETR {xml_file}', f.write, blocksize=1024*1024)
                print(f"Downloaded {xml_file}")
            # Decompress XML 
            with gzip.open(local_gz_path, 'rb') as f_in, open(local_xml_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed {xml_file}")
            break  # success, exit retry loop
        except Exception as e:
            print(f"Attempt {attempt} failed for {xml_file}: {e}")
        attempt += 1
        
    if attempt == 3:
        print(f"Skipping {xml_file} after 2 failed attempts")
        continue

    # Parse XML and write JSONL
    with open(local_xml_path, 'r', encoding='utf-8') as f_in, open(local_jsonl_path, 'w', encoding='utf-8') as f_out:
        tree = ET.parse(f_in)
        root = tree.getroot()
        for article in root.findall(".//PubmedArticle"):
            pmid_el = article.find(".//PMID")
            title_el = article.find(".//ArticleTitle")
            abstract_el = article.find(".//Abstract/AbstractText")
            # Extract all Mesh tags
            mesh_terms = [mh.find("DescriptorName").text 
                        for mh in article.findall(".//MeshHeadingList/MeshHeading") 
                        if mh.find("DescriptorName") is not None]
            record = {
                "pmid": pmid_el.text if pmid_el is not None else "",
                "title": title_el.text if title_el is not None else "",
                "abstract": abstract_el.text if abstract_el is not None else "",
                "mesh_terms": mesh_terms
            }
            json.dump(record, f_out)
            f_out.write("\n")
    print(f"Converted {xml_file} -> {local_jsonl_path}")

    # Remove original files
    os.remove(local_xml_path)
    print(f"Removed file: {local_xml_path}")

ftp.quit()
print("All files downloaded and converted to JSONL.")
