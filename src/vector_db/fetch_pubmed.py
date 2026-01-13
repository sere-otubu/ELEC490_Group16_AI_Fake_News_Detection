from Bio import Entrez
import os

Entrez.email = "21is8@queensu.ca"  # Required by NCBI
DATA_DIR = "data/pubmed_api"
os.makedirs(DATA_DIR, exist_ok=True)

def search_med_articles(query, max_results=100):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_details(id_list):
    ids = ",".join(id_list)
    handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
    return Entrez.read(handle)

# Search for 100 articles about "vaccine safety"
print("Searching PubMed...")
ids = search_med_articles("vaccine safety", max_results=100)
papers = fetch_details(ids)

for paper in papers['PubmedArticle']:
    try:
        article = paper['MedlineCitation']['Article']
        title = article['ArticleTitle']
        abstract = article['Abstract']['AbstractText'][0]
        pmid = paper['MedlineCitation']['PMID']
        
        with open(f"{DATA_DIR}/pmid_{pmid}.txt", "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\n\nAbstract: {abstract}")
    except:
        continue

print("Download complete.")