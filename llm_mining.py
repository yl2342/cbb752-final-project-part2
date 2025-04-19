#!/usr/bin/env python3

import json
import sys
import os
import argparse
import glob
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv

def load_abstracts(json_file):
    """Load the abstracts from a single JSON file associated with a gene"""
    try:
        with open(json_file, 'r') as f:
            articles = json.load(f)
        return articles
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return []

def load_all_abstracts(abstracts_dir="pubmed_abstracts"):
    """Load all abstracts from JSON files in the directory"""
    if not os.path.exists(abstracts_dir):
        print(f"Error: Directory {abstracts_dir} does not exist")
        return {}
        
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(abstracts_dir, "*_pubmed_abstracts.json"))
    
    if not json_files:
        print(f"No JSON files found in {abstracts_dir}")
        return {}
        
    print(f"Found {len(json_files)} JSON files in {abstracts_dir}")
    
    # Dictionary to store gene name -> articles mapping
    gene_abstracts = {}
    
    # Process each JSON file
    for json_file in json_files:
        # Extract gene name from filename
        filename = os.path.basename(json_file)
        gene = filename.split('_pubmed_abstracts.json')[0]
        
        # Load articles for this gene
        abstracts = load_abstracts(json_file)
        
        if abstracts:
            gene_abstracts[gene] = abstracts
            print(f"Loaded {len(abstracts)} abstracts for gene {gene}")
        
    return gene_abstracts

def create_prompt_template(gene_abstracts):
    """
    Create a structured prompt template for large language models to analyze
    the abstracts for key biological information about the genes
    """
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Count total abstracts
    total_abstracts = sum(len(abstracts) for abstracts in gene_abstracts.values())
    total_genes = len(gene_abstracts)
    
    # Start building the prompt
    prompt = f"""# Systematic Analysis of the pubmed literature for prioritized genes
    
## Background
Below are {total_abstracts} scientific abstracts from PubMed about {total_genes} different genes, retrieved on {current_date}. 
These abstracts represent recent and relevant research on these genes.

## Abstracts

"""

    # Add each abstract with its metadata organized by gene
    for gene, abstracts in gene_abstracts.items():
        prompt += f"# Gene: {gene} ({len(abstracts)} abstracts)\n"
        
        for i, abstract_data in enumerate(abstracts, 1):
            # Check if abstract_data is a dictionary, if not, skip it or handle appropriately
            if not isinstance(abstract_data, dict):
                print(f"Warning: Abstract {i} for gene {gene} is not in the expected format. Skipping...")
                continue
                
            title = abstract_data.get('title', 'No Title')
            journal = abstract_data.get('journal', 'Unknown Journal')
            date = abstract_data.get('date', 'Unknown Date')
            abstract = abstract_data.get('abstract', 'No abstract available')
            pmid = abstract_data.get('pmid', 'Unknown PMID')
            
            prompt += f"### {gene} - Abstract {i}: {title}\n"
            prompt += f"<Journal>: {journal}\n"
            prompt += f"<Date>: {date}\n"
            prompt += f"<PMID>: {pmid}\n"
            prompt += f"<Content>: {abstract}\n"
            prompt += "---\n"
    
    # Add the analysis instructions
    prompt += f"""## Analysis Instructions

You are an outstanding and meticulous biomedical informatics researcher. 
Based on these {total_abstracts} abstracts about {total_genes} different genes, please provide a comprehensive analysis with the following components:

1. Frequent Biological Terms: Identify and list the most frequently mentioned biological terms across all abstracts, organized by gene.

2. Key Findings: Summarize the major discoveries and findings about each gene. Highlight consensus findings that appear in multiple abstracts.

3. Gene-Disease Associations: Identify all diseases, disorders, or pathological conditions associated with each gene.

4. Correlation Analysis: Identify correlations between specific terms appearing together within and across publications.

5. Disease Implications: Discuss the implications for disease mechanisms, progression, or treatments based on the discovered correlations.

6. Gene Interactions and Pathways: Identify any interactions or shared pathways between the different genes based on the literature.


Once you have done the analysis, please organize your findings and provide an overall summary of the collective insights that emerge from studying these genes together.
Please do it in a way that is structured, clear, concise, easy to understand and follow. 

Then do a search for these genes on comprehensive databases such as UniProt or GeneCards, compare your findings with protein function annotations from these databases. 
Highlight any consistencies or discrepancies identified in this comparative analysis and discuss their significance.

Please structure your response in a simple format that is compatible with the MS word document (Do not use markdown or other decorations for text).
Limit your response within 800 words, and in a way that can easily fit into an academic paper/essay as a section.
You can use bullet points or tables to organize your findings to make it more readable. If you need to refer to a specific article, please use the PMID.
Also make sure to include the total number of genes and the total number of abstracts analyzed at the beginning of your response.

"""
    
    return prompt

def save_prompt(prompt, output_dir="prompts", filename="integrated_prompt.txt"):
    """Save the generated prompt to a text file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, filename)
    
    with open(output_file, 'w') as f:
        f.write(prompt)
    
    print(f"Saved analysis prompt to {output_file}")
    return output_file

def get_gemini_response(prompt, api_key):
    """Send the prompt to Google's Gemini API and get a response"""
    print("Sending prompt to Gemini API...")
    
    # Configure the API with the user's key
    genai.configure(api_key=api_key)
    
    # Configure generation parameters
    generation_config = {
        "temperature": 0.05,  # Low temperature for more factual responses
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 32768,  # Get a comprehensive response
    }
    
    # Use Gemini Pro model
    model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25', 
                                generation_config=generation_config)
    
    # Generate response
    response = model.generate_content(prompt)
    
    print("Response received from Gemini")
    return response.text

def save_response(response, output_dir="llm_responses", filename="part2_text_mining_response.txt"):
    """Save the LLM response to a file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, filename)
    
    with open(output_file, 'w') as f:
        f.write(response)
    
    print(f"Saved analysis response to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate text mining analysis for prioritized genes from PubMed abstracts using Gemini API')
    parser.add_argument('--check', '-c', action='store_true', help='Only generate and save prompt without calling the API (for sanity check)')
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Load all abstracts from all genes
    gene_abstracts = load_all_abstracts(abstracts_dir="pubmed_abstracts")
    
    if not gene_abstracts:
        print("No abstracts found. Please run pubmed_gene_search.py first.")
        sys.exit(1)
    
    # Get total article count
    total_abstracts = sum(len(abstracts) for abstracts in gene_abstracts.values())
    
    # Create the prompt
    prompt = create_prompt_template(gene_abstracts)
    
    # Save the prompt
    prompt_file = save_prompt(prompt, output_dir="prompts", filename="integrated_prompt.txt")
    
    print(f"Generated analysis prompt for {len(gene_abstracts)} genes with {total_abstracts} total abstracts")
    
    # If check flag is set, stop here
    if args.check:
        print("Check flag is set. Stopping after prompt generation. Review the prompt file before proceeding.")
        sys.exit(0)
    
    # Get API key ( .env file > environment variable)
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        print("Error: Gemini API key not found.")
        print("Please create a .env file with GEMINI_API_KEY=your_key")
        sys.exit(1)
        
    # Get response from Gemini
    response = get_gemini_response(prompt, api_key)
    
    # Save the response
    response_file = save_response(response, output_dir="llm_responses", filename="part2_text_mining_response.txt")
    
    print(f"Analysis complete. Prompt saved to {prompt_file} and response saved to {response_file}")

if __name__ == "__main__":
    main() 