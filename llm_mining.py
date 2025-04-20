#!/usr/bin/env python3

import json
import sys
import os
import argparse
import glob
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import logging # Import logging module

def load_pmc_articles_per_gene(json_file):
    """Load the articles (metadata + full text) from the JSON file of a single gene."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:  # Specify encoding
            articles = json.load(f)
        return articles
    except json.JSONDecodeError as e:
        # Use logging for errors
        logging.error(f"Error decoding JSON from file {json_file}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error loading JSON file {json_file}: {e}")
        return []

def load_all_pmc_articles(articles_dir="pmc_full_texts"):
    """Load all articles from PMC JSON files in the directory."""
    if not os.path.exists(articles_dir):
        logging.error(f"Directory '{articles_dir}' does not exist. Please run pmc_fulltext_fetcher.py first.")
        return {}

    # Get all JSON files matching the pattern
    json_files = glob.glob(os.path.join(articles_dir, "*_pmc_articles.json"))

    if not json_files:
        logging.warning(f"No PMC article JSON files found in '{articles_dir}'. Pattern: *_pmc_articles.json")
        return {}

    logging.info(f"Found {len(json_files)} PMC article JSON files in '{articles_dir}'")

    # Dictionary to store gene name -> articles mapping
    gene_articles_data = {}

    # Process each JSON file
    # each json file contains the articles for a single gene
    for json_file in json_files:
        # Extract gene name from filename
        filename = os.path.basename(json_file)
        # Handle potential variations if needed, assumes format GENE_pmc_articles.json
        try:
            gene = filename.split('_pmc_articles.json')[0]
        except IndexError:
             logging.warning(f"Could not extract gene name from filename '{filename}'. Skipping.")
             continue

        # Load articles for this gene
        articles = load_pmc_articles_per_gene(json_file)

        if articles:
            gene_articles_data[gene] = articles
            logging.info(f"Loaded {len(articles)} articles for gene '{gene}' from '{filename}'")
        else:
             logging.warning(f"No articles loaded for gene '{gene}' from file '{filename}'")

    return gene_articles_data

def create_analysis_prompt(gene_articles_data):
    """
    Create a structured prompt template for LLMs to analyze the full texts
    for key biological information about the genes.
    """
    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Count total articles and genes
    total_articles = sum(len(articles) for articles in gene_articles_data.values())
    total_genes = len(gene_articles_data)

    # --- Prompt Part 1: Background --- 
    prompt_part1 = f"""# Systematic Analysis of PMC Open Access Literature for Prioritized Genes

## Background
Below are {total_articles} full-text scientific articles from the PubMed Central (PMC) Open Access Subset, retrieved around {current_date}. 
These articles cover {total_genes} different genes and represent recent and relevant research.

"""

    # --- Prompt Part 2: Instructions --- 
    prompt_part2 = f"""## Analysis Instructions

You are an expert biomedical informatics researcher specializing in text mining of scientific literature.
Based *solely* on the {total_articles} full-text articles provided below*concerning {total_genes} different genes, 
please perform a comprehensive analysis focusing on **human-relevant** biological information and provide a academic report covering the following two sections:

1. Overall Summary based on the provided articles, which should:
* Systematically extract the most frequent and important (top 10) biological terms, key findings, and gene-disease associations across all the publications. Do not separate the analysis based on the genes or articles. If references to specific articles are made, make sure to use PMCID .
* Identify correlations between specific terms within and across publications. 
* Discuss the implications for disease based on these correlations. 
* Should be clear and concise, less than 500 words.

2. Comparative Analysis with External Databases, which should:
* Perform a separate search for these genes on comprehensive databases (like UniProtKB, GeneCards, OMIM). 
* Compare your summary based on the provided articles with protein function annotations from comprehensive databases such as UniProt or GeneCards. 
* Highlight any consistencies or discrepancies identified in this comparative analysis and discuss their significance.
* Should be clear and concise, less than 500 words.


**Other Output Requirements:**

*   **Format:** Structure your report clearly, suitable for inclusion in a report (e.g., use headings, bullet points). Avoid Markdown formatting like bolding or italics if possible, aiming for plain text compatibility.
*   **Word Count:** Aim for the total report to be around 500-1000 words, focusing on clarity and conciseness.
*   **Referencing:** Only refer to specific information when necessary. When need to refer to specific information, cite the corresponding PMCID.
*   **Header:** Begin your response by stating the total number of genes and total number of full-text articles analyzed.
"""

    # --- Prompt Part 3: Articles --- 
    prompt_part3 = "## Full-Text Articles\n\n"

    # Add each article with its metadata organized by gene
    for gene, articles in gene_articles_data.items():
        prompt_part3 += f"### Gene: {gene} ({len(articles)} articles)\n\n"

        for i, article_data in enumerate(articles, 1):
            if not isinstance(article_data, dict):
                logging.warning(f"Article {i} for gene {gene} is not in the expected dictionary format. Skipping...")
                continue

            title = article_data.get('title', 'No Title Provided')
            date = article_data.get('publish_date', 'Unknown Date')
            full_text = article_data.get('full_text', 'No full text available')
            pmcid = article_data.get('pmcid', 'Unknown PMCID')

            prompt_part3 += f"#### {gene} - Article {i}: {title}\n"
            prompt_part3 += f"<Date>: {date}\n"
            prompt_part3 += f"<PMCID>: {pmcid}\n"
            prompt_part3 += f"<Full Text Content>:\n{full_text}\n"
            prompt_part3 += "---\n"
        prompt_part3 += "\n"

    # Combine the parts
    full_prompt = prompt_part1 + prompt_part2 + prompt_part3

    return full_prompt

def save_prompt(prompt, output_dir="prompts", filename="fulltext_analysis_prompt.txt"):
    """Save the generated prompt to a text file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, filename)

    with open(output_file, 'w', encoding='utf-8') as f:  # Specify encoding
        f.write(prompt)

    logging.info(f"Saved analysis prompt to {output_file}")
    return output_file

def get_gemini_response(prompt, api_key, model_name='gemini-2.0-flash'):
    """Send the prompt to Google's Gemini API and get a response using the specified model"""
    logging.info(f"Sending prompt to Gemini API using model: {model_name}...")

    # Configure the API with the user's key
    genai.configure(api_key=api_key)

    # Configure generation parameters
    generation_config = {
        "temperature": 0.1,  
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 65586, 
    }

    # Use the specified Gemini model
    model = genai.GenerativeModel(model_name,
                                generation_config=generation_config)

    try:
        # Generate response
        response = model.generate_content(prompt)
        logging.info("Response received from Gemini")
        # Accessing the text part safely
        if response and response.parts:
             return response.text
        else:
            logging.warning("Gemini response structure might be unexpected or empty.")
            logging.debug(f"Full response object: {response}")
            if hasattr(response, 'candidates') and response.candidates:
                try:
                    return response.candidates[0].content.parts[0].text
                except (IndexError, AttributeError) as e:
                    logging.error(f"Could not extract text from candidate: {e}")
                    return None
            return None

    except Exception as e:
        logging.error(f"An error occurred while calling the Gemini API with model {model_name}: {e}")
        return None

def save_response(response_text, output_dir="llm_responses", filename="part2_fulltext_mining_response.txt"):
    """Save the LLM response text to a file"""
    if not response_text:
        logging.warning("No response text provided to save.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, filename)

    with open(output_file, 'w', encoding='utf-8') as f:  # Specify encoding
        f.write(response_text)

    logging.info(f"Saved analysis response to {output_file}")
    return output_file

def main():
    
    # --- Logging Setup --- 
    log_dir = 'logs' # Define log directory name
    
    # Create timestamp for log file name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_filename = f"llm_mining_{timestamp}.log" # Construct filename with timestamp
    log_file = os.path.join(log_dir, log_filename) # Define full log file path

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, # Set default logging level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file), # Log to timestamped file inside 'logs' directory
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )
    logging.info(f"Starting LLM Mining script... Logging to {log_file}") # Log the filename
    # --- End Logging Setup ---

    parser = argparse.ArgumentParser(description='Generate text mining analysis for genes from PMC full texts using Gemini API')
    parser.add_argument('--check', '-c', action='store_true', help='Only generate and save prompt without calling the API (for sanity check)')
    parser.add_argument('--input_dir', '-i',
                        default="pmc_full_texts", # Default input dir
                        help='Directory containing the PMC article JSON files (default: pmc_full_texts).')
    parser.add_argument('--model', '-m',
                        default='gemini-2.0-flash', # Set a default model
                        help='Name of the Gemini model to use (default: gemini-2.0-flash).') 

    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")

    # Load environment variables from .env file
    load_dotenv()
    logging.info(".env file loaded (if exists).")

    # Load all articles from PMC JSON files
    logging.info(f"Loading PMC articles from directory: '{args.input_dir}'")
    gene_articles_data = load_all_pmc_articles(articles_dir=args.input_dir)

    if not gene_articles_data:
        logging.error(f"No articles found in '{args.input_dir}'. Please ensure pmc_fulltext_fetcher.py has run successfully.")
        sys.exit(1)

    # Get total article count
    total_articles = sum(len(articles) for articles in gene_articles_data.values())
    logging.info(f"Total articles loaded: {total_articles} across {len(gene_articles_data)} genes.")

    # Create the prompt
    logging.info("Generating analysis prompt...")
    prompt = create_analysis_prompt(gene_articles_data)
    logging.info("Prompt generation complete.")

    # --- Calculate and Print Token Count --- 
    api_key = os.environ.get('GEMINI_API_KEY')
    token_count = None
    if api_key:
        try:
            logging.info(f"Calculating prompt token count for model: {args.model}...") # Use args.model
            genai.configure(api_key=api_key)
            # Use the specified model for accurate counting
            token_model = genai.GenerativeModel(args.model) # Use args.model here
            token_count_response = token_model.count_tokens(prompt)
            token_count = token_count_response.total_tokens
            logging.info(f"---> Estimated token count for the generated prompt: {token_count} <--- ({args.model})") # Show model used
        except Exception as e:
            logging.warning(f"Could not count tokens using model {args.model}. Error: {e}") # Show model used
    else:
        logging.error("Gemini API key not found. Cannot count tokens or proceed.") 
        print("Error: Gemini API key not found.") 
        print("Please create a .env file with GEMINI_API_KEY=your_key")
        sys.exit(1)
    # --- End Token Counting ---

    # Save the prompt
    prompt_file = save_prompt(prompt, output_dir="prompts", filename="fulltext_analysis_prompt.txt")

    logging.info(f"Generated analysis prompt for {len(gene_articles_data)} genes with {total_articles} total articles")

    # If check flag is set, stop here
    if args.check:
        logging.info("Check flag is set. Stopping after prompt generation.")
        if token_count is not None:
             logging.info(f"(Model: {args.model}, Prompt token count: {token_count})") # Show model used
        sys.exit(0)

    # --- Confirmation before API Call --- 
    logging.warning("--- CONFIRMATION REQUIRED BEFORE API CALL ---")
    logging.warning(f"Model to be used: {args.model}")
    logging.warning(f"Calculated Input Tokens: {token_count}")
    # Note: We can't pre-calculate exact output tokens, but we know the configured max limit
    # Retrieve max_output_tokens from where it's defined (or hardcode if consistent)
    # For simplicity, let's assume it's 65586 as set in get_gemini_response
    max_output_tokens_config = 65586 # Value from generation_config in get_gemini_response
    logging.warning(f"Maximum Output Tokens Configured: {max_output_tokens_config}") 
    logging.warning("Please review model capabilities, pricing, and potential rate limits, especially for free tiers:")
    logging.warning("Reference: https://ai.google.dev/gemini-api/docs/models")
    
    try:
        confirm = input("Proceed with sending prompt to the Gemini API? (y/n): ")
        if confirm.lower().strip() != 'y':
            logging.info("User cancelled operation. Exiting.")
            sys.exit(0)
        else:
            logging.info("User confirmed. Proceeding with API call...")
    except EOFError: # Handle cases where input is piped or unavailable
        logging.error("Could not get user confirmation (EOFError). Exiting.")
        sys.exit(1)
    # --- End Confirmation --- 

    # Get response from Gemini - pass the model name
    response_text = get_gemini_response(prompt, api_key, args.model)

    # Save the response
    if response_text:
        response_file = save_response(response_text, output_dir="llm_responses", filename="part2_fulltext_mining_response.txt")
        logging.info(f"Analysis complete. Prompt saved to {prompt_file} and response saved to {response_file}")
    else:
        logging.error("Failed to get a valid response from the Gemini API. No response file saved.")
        sys.exit(1)

    logging.info("LLM Mining script finished successfully.")

if __name__ == "__main__":
    main() 