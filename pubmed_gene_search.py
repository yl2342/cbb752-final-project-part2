#!/usr/bin/env python3

import sys
import json
import csv
import os
import argparse
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.getenv("EMAIL")

def search_pubmed(gene):
    """
    Search PubMed for top 10 articles that have the gene name in their title
    Returns all matching results
    """
    print(f"Searching PubMed for articles with '{gene}' in the title...")
    
    # Search PubMed for articles containing the gene name in the title
    try:
        # Construct a query for articles with gene in title
        search_query = f"{gene}[Title] AND human[Organism]"
        
        # First get the IDs of articles related to this gene - no limit on results
        search_handle = Entrez.esearch(db="pubmed", 
                                       term=search_query, 
                                       retmax=10,  # Increased to get more results
                                       sort="relevance")
        search_results = Entrez.read(search_handle)
        search_handle.close()
        
        if not search_results["IdList"]:
            print(f"No articles found with '{gene}' in the title")
            return
            
        article_ids = search_results["IdList"]
        total_articles = len(article_ids)
        print(f"Found {total_articles} articles with '{gene}' in the title")
        
        # Then fetch the details for these articles
        fetch_handle = Entrez.efetch(db="pubmed", 
                                    id=article_ids, 
                                    rettype="medline", 
                                    retmode="text")
        articles = fetch_handle.read()
        fetch_handle.close()
        
        # Process articles
        processed_articles = []
        current_article = {}
        current_field = None
        
        for line in articles.splitlines():
            if not line.strip():
                continue
                
            # Check if this is a new field
            if line[:4].strip() and not line.startswith("      "):
                current_field = line[:4].strip()
                
                if current_field == "PMID":
                    if current_article:
                        processed_articles.append(current_article)
                        current_article = {}
                    current_article["pmid"] = line.replace("PMID- ", "")
                elif current_field == "TI":
                    current_article["title"] = line.replace("TI  - ", "")
                elif current_field == "AB":
                    current_article["abstract"] = line.replace("AB  - ", "")
                elif current_field == "AU":
                    if "authors" not in current_article:
                        current_article["authors"] = []
                    current_article["authors"].append(line.replace("AU  - ", ""))
                elif current_field == "DP":
                    current_article["date"] = line.replace("DP  - ", "")
                elif current_field == "JT":
                    current_article["journal"] = line.replace("JT  - ", "")
            else:
                # This is a continuation of the previous field
                if current_field == "TI" and "title" in current_article:
                    current_article["title"] += " " + line.strip()
                elif current_field == "AB" and "abstract" in current_article:
                    current_article["abstract"] += " " + line.strip()
                    
        # Add the last article if it exists
        if current_article:
            processed_articles.append(current_article)
        
        # Double check that each article has the gene name in the title (case insensitive)
        filtered_articles = []
        for article in processed_articles:
            title = article.get("title", "").upper()
            if gene.upper() in title:
                filtered_articles.append(article)
        
        print(f"Successfully processed {len(filtered_articles)} articles")
        return filtered_articles
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def save_to_json(articles, gene, output_dir="pubmed_abstracts"):
    """Save the articles to a JSON file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, f"{gene}_pubmed_abstracts.json")
    
    with open(output_file, 'w') as f:
        json.dump(articles, f, indent=2)
    
    print(f"Saved {len(articles)} articles to {output_file}")
    return output_file

def read_gene_list(file_path):
    """Read gene names list from a text file, one gene per line"""
    genes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene:  # Skip empty lines
                    genes.append(gene)
        return genes
    except Exception as e:
        print(f"Error reading gene file: {e}")
        return None

def process_single_gene(gene):
    """Process a single gene and save the results"""
    articles = search_pubmed(gene)
    
    if articles and len(articles) > 0:
        save_to_json(articles, gene)
        return len(articles)
    else:
        print(f"No articles found or processed for {gene}")
        return 0

def main():
    parser = argparse.ArgumentParser(description='Search PubMed for articles with gene names in the title')
    
    # Create a mutually exclusive group for gene input method
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--gene', '-g', help='Single gene name to search for')
    input_group.add_argument('--list', '-l', help='Path to text file with gene names list (one per line)')
    
    args = parser.parse_args()
    
    if args.gene:
        # Process a single gene
        total_abstracts = process_single_gene(args.gene)
        print(f"\nTotal abstracts retrieved: {total_abstracts}")
    elif args.list:
        # Process multiple genes from a list file
        genes = read_gene_list(args.list)
        
        if not genes:
            print("No genes found in the file or file could not be read.")
            sys.exit(1)
            
        print(f"Found {len(genes)} genes in the file. Processing sequentially...")
        
        success_count = 0
        total_abstracts = 0
        for i, gene in enumerate(genes, 1):
            print(f"\n[{i}/{len(genes)}] Processing gene: {gene}")
            abstracts_count = process_single_gene(gene)
            if abstracts_count > 0:
                success_count += 1
                total_abstracts += abstracts_count
                
        print(f"\nCompleted processing {len(genes)} genes. Successfully retrieved {total_abstracts} abstracts for {success_count} genes.")
        print(f"Total abstracts retrieved across all genes: {total_abstracts}")

if __name__ == "__main__":
    main() 