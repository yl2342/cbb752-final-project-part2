#!/usr/bin/env python3

import json
import sys
import os
import argparse
import glob
from datetime import datetime
import google.genai as genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, AutomaticFunctionCallingConfig

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

def create_overall_summary_prompt(gene_articles_data):
    """
    Create a structured prompt template for LLMs to analyze the full texts
    for key biological information about the genes - Overall Summary part only.
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
Based *solely* on the {total_articles} full-text articles provided below concerning {total_genes} different genes, 
please perform a comprehensive analysis focusing on **human-relevant** biological information and provide an academic report covering the following:

1. Overall Summary based on the provided articles, which should:
* Systematically extract the most frequent and important (top 5) biological terms, key findings, and gene-disease associations across all the publications. Do not separate the analysis based on the genes or articles. If references to specific articles are made, make sure to use PMCID.
* Identify correlations between specific terms within and across publications. 
* Discuss the implications for disease based on these correlations. 
* Should be clear and concise, less than 500 words.

**Output Requirements:**

*   **Format:** Structure your report clearly, suitable for inclusion in a report (e.g., use headings, bullet points). Avoid Markdown formatting like bolding or italics if possible, aiming for plain text compatibility.
*   **Word Count:** Aim for the total report to be around 500 words, focusing on clarity and conciseness.
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

def create_comparative_analysis_prompt(gene_articles_data, overall_summary):
    """
    Create a structured prompt template for LLMs to perform comparative analysis
    based on the previously generated overall summary.
    """
    # Count total articles and genes
    total_articles = sum(len(articles) for articles in gene_articles_data.values())
    total_genes = len(gene_articles_data)

    # List of gene names for reference
    gene_names = list(gene_articles_data.keys())
    
    prompt = f"""# Comparative Analysis with External Databases for Gene Data

## Background
Previously {total_articles} full-text scientific articles covering {total_genes} different genes 
({', '.join(gene_names)}) were analyzed and produced the following overall summary:

-----
{overall_summary}
-----

## Analysis Instructions

You are an expert biomedical informatics researcher specializing in text mining of scientific literature.
Based on the overall summary above, please perform a comparative analysis with external databases as follows:

1. Comparative Analysis with External Databases UniProt, which should:
* Perform an external Google search to use UniProt as an external database to search for these genes: {', '.join(gene_names)}. Limit your remote calls to only search for these genes in UniProt.
* Compare the previous overall summary with protein function annotations and other information retrieved from the search on UniProt. 
* Highlight any consistencies or discrepancies identified in this comparative analysis and discuss their significance.
* Should be clear and concise, less than 500 words.

**Output Requirements:**
*   **Format:** Structure your report clearly, suitable for inclusion in a report (e.g., use headings, bullet points). Avoid Markdown formatting like bolding or italics if possible, aiming for plain text compatibility.
*   **Word Count:** Aim for the report to be around 500 words, focusing on clarity and conciseness.
"""

    return prompt

def save_prompt(prompt, filename, output_dir="prompts"):
    """Save the generated prompt to a text file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, filename)

    with open(output_file, 'w', encoding='utf-8') as f:  # Specify encoding
        f.write(prompt)

    logging.info(f"Saved analysis prompt to {output_file}")
    return output_file

def initalize_gemini_client(api_key):
    """Initialize the Gemini client"""
    client = genai.Client(api_key=api_key) 
    return client

def get_gemini_response(client, prompt, model_name='gemini-2.0-flash', max_output_tokens= 16384, use_grounding_tools=False):
    """Send the prompt to Google's Gemini API and get a response using the specified model, with optional tools"""
    logging.info(f"Sending prompt to Gemini API using model: {model_name}...")

    # Configure tools if requested
    tools = []
    if use_grounding_tools:
        # set up the grounding tool: google search
        google_search_tool = Tool(
            google_search=GoogleSearch()
        )
        tools = [google_search_tool]
        logging.info(f"Set up API and grounding tool (Google Search) for: {model_name}")
    else:
        logging.info(f"Set up API without grounding tools for: {model_name}")
    
    try:
        # Generate response
        response = client.models.generate_content(
            model = model_name,
            contents = prompt,
            config = GenerateContentConfig(
                tools = tools,
                temperature = 0,
                max_output_tokens = max_output_tokens,
                automatic_function_calling = AutomaticFunctionCallingConfig(maximum_remote_calls=10) if use_grounding_tools else None
            )
        )
        logging.info("Response received from Gemini API")
        
        # Check if response is empty
        if response:
            # Save the raw response object as JSON for debugging purposes
            debug_dir = "debug"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Add timestamp to debug filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_file = os.path.join(debug_dir, f"raw_response_{'with_grounding_tools' if use_grounding_tools else 'no_grounding_tools'}_{timestamp}.json")
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    # Just save the raw response as is
                    json.dump(response, f, indent=2, default=str)
                logging.info(f"Saved raw response object to {debug_file} for debugging")
            except Exception as e:
                logging.warning(f"Failed to save raw response for debugging: {e}")
            
            return response
        else:
            logging.warning("Gemini response is empty.")
            return None

    except Exception as e:
        logging.error(f"An error occurred while calling the Gemini API with model {model_name}: {e}")
        return None

def save_response(response, output_dir="llm_responses", 
                  response_filename="overall_summary.txt",
                  grounding_filename=None):
    """Save the LLM response text to a file and optionally grounding metadata to a JSON file"""
    if not response:
        logging.warning("No response provided to save.")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the response text - collect text from all parts
    response_file_path = os.path.join(output_dir, response_filename)
    
    # Extract text from all parts of the response
    all_parts_text = []
    for part in response.candidates[0].content.parts:
        if hasattr(part, 'text') and part.text:
            all_parts_text.append(part.text)
    
    # Concatenate all parts with a newline between them
    response_text = "\n\n".join(all_parts_text)
    
    with open(response_file_path, 'w', encoding='utf-8') as f:  # Specify encoding
        f.write(response_text)
    
    logging.info(f"Saved analysis response with {len(all_parts_text)} parts to {response_file_path}")
    
    # Save the grounding metadata to a JSON file if requested
    grounding_file_path = None
    if grounding_filename and hasattr(response.candidates[0], 'grounding_metadata'):
        grounding_file_path = os.path.join(output_dir, grounding_filename)
        grounding_data = response.candidates[0].grounding_metadata
        
        try:
            with open(grounding_file_path, 'w', encoding='utf-8') as f:
                # Just save the raw grounding metadata as is
                json.dump(grounding_data, f, indent=2, default=str)
            logging.info(f"Saved grounding metadata to {grounding_file_path}")
        except Exception as e:
            logging.warning(f"Failed to save grounding metadata: {e}")
    
    return response_file_path, grounding_file_path

def load_response_text(response_file):
    """Load the text content of a previously saved response"""
    try:
        with open(response_file, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load response file {response_file}: {e}")
        return None

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

    # Connect to the Gemini API and initialize the client
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logging.error("Gemini API key not found. Please create a .env file with GEMINI_API_KEY=your_key.")
        sys.exit(1)

    client = initalize_gemini_client(api_key)
    
    # Define file paths for intermediate results
    output_dir = "llm_responses"
    summary_file = os.path.join(output_dir, "overall_summary.txt")
    
    # ======== STEP 1: Overall Summary =========
    # Create the overall summary prompt
    logging.info("Generating overall summary prompt...")
    summary_prompt = create_overall_summary_prompt(gene_articles_data)
    logging.info("Overall summary prompt generation complete.")
    
    # Save the prompt
    summary_prompt_file = save_prompt(summary_prompt, filename="overall_summary_prompt.txt", output_dir="prompts")

    # Calculate token count for summary prompt
    try:
        logging.info(f"Calculating summary prompt token count for model: {args.model}...")
        token_count_response = client.models.count_tokens(
            model = args.model,
            contents = summary_prompt
        )
        summary_token_count = token_count_response.total_tokens
        logging.info(f"---> Estimated token count for the summary prompt: {summary_token_count} <--- ({args.model})")
    except Exception as e:
        logging.warning(f"Could not count tokens using model {args.model}. Error: {e}")
        summary_token_count = "unknown"

    # --- Confirmation before first API Call --- 
    logging.warning("--- CONFIRMATION REQUIRED BEFORE FIRST API CALL (OVERALL SUMMARY) ---")
    logging.warning(f"Model to be used: {args.model}")
    logging.warning(f"Calculated Input Tokens: {summary_token_count}")
    max_output_tokens_config = 16384
    logging.warning(f"Maximum Output Tokens Configured: {max_output_tokens_config}")
    
    try:
        confirm = input(f"To proceed, retype the model name ({args.model}) or enter a different model name: ")
        if not confirm.strip():
            logging.info("No model name entered. Exiting.")
            sys.exit(0)
        elif confirm.strip() != args.model:
            logging.info(f"Model changed from {args.model} to {confirm.strip()}")
            args.model = confirm.strip()
        logging.info(f"User confirmed. Proceeding with first API call using model: {args.model}...")
    except EOFError:
        logging.error("Could not get user confirmation (EOFError). Exiting.")
        sys.exit(1)

    # Get first response from Gemini (WITHOUT tools)
    summary_response = get_gemini_response(
        client=client, 
        prompt=summary_prompt, 
        model_name=args.model, 
        max_output_tokens=max_output_tokens_config,
        use_grounding_tools=False
    )

    # Save the first response
    if summary_response:
        summary_response_file, _ = save_response(
            summary_response, 
            output_dir=output_dir, 
            response_filename="overall_summary.txt"
        )
        logging.info(f"Overall summary complete. Prompt saved to {summary_prompt_file} and response saved to {summary_response_file}")
    else:
        logging.error("Failed to get a valid response for overall summary. Exiting.")
        sys.exit(1)

    # ======== STEP 2: Comparative Analysis =========
    # Load the overall summary
    overall_summary = load_response_text(summary_file)
    if not overall_summary:
        logging.error(f"Failed to load overall summary from {summary_file}. Cannot proceed with comparative analysis.")
        sys.exit(1)
    
    # Create the comparative analysis prompt
    logging.info("Generating comparative analysis prompt...")
    comparative_prompt = create_comparative_analysis_prompt(gene_articles_data, overall_summary)
    logging.info("Comparative analysis prompt generation complete.")
    
    # Save the prompt
    comparative_prompt_file = save_prompt(comparative_prompt, filename="comparative_analysis_prompt.txt", output_dir="prompts")

    # Calculate token count for comparative prompt
    try:
        logging.info(f"Calculating comparative prompt token count for model: {args.model}...")
        token_count_response = client.models.count_tokens(
            model = args.model,
            contents = comparative_prompt
        )
        comparative_token_count = token_count_response.total_tokens
        logging.info(f"---> Estimated token count for the comparative prompt: {comparative_token_count} <--- ({args.model})")
    except Exception as e:
        logging.warning(f"Could not count tokens using model {args.model}. Error: {e}")
        comparative_token_count = "unknown"

    # --- Confirmation before second API Call --- 
    logging.warning("--- CONFIRMATION REQUIRED BEFORE SECOND API CALL (COMPARATIVE ANALYSIS) ---")
    logging.warning(f"Model to be used: {args.model}")
    logging.warning(f"Calculated Input Tokens: {comparative_token_count}")
    logging.warning(f"Maximum Output Tokens Configured: {max_output_tokens_config}")
    logging.warning("This call will use Google Search tools for UniProt data retrieval.")
    
    try:
        confirm = input(f"To proceed, retype the model name ({args.model}) or enter a different model name: ")
        if not confirm.strip():
            logging.info("No model name entered. Exiting.")
            sys.exit(0)
        elif confirm.strip() != args.model:
            logging.info(f"Model changed from {args.model} to {confirm.strip()}")
            args.model = confirm.strip()
        logging.info(f"User confirmed. Proceeding with second API call using model: {args.model}...")
    except EOFError:
        logging.error("Could not get user confirmation (EOFError). Exiting.")
        sys.exit(1)

    # Get second response from Gemini (WITH tools)
    comparative_response = get_gemini_response(
        client=client, 
        prompt=comparative_prompt, 
        model_name=args.model, 
        max_output_tokens=max_output_tokens_config,
        use_grounding_tools=True
    )

    # Save the second response
    if comparative_response:
        comparative_response_file, grounding_file = save_response(
            comparative_response, 
            output_dir=output_dir, 
            response_filename="comparative_analysis.txt",
            grounding_filename="comparative_analysis_grounding.json"
        )
        logging.info(f"Comparative analysis complete. Prompt saved to {comparative_prompt_file} and response saved to {comparative_response_file}")
        if grounding_file:
            logging.info(f"Grounding metadata saved to {grounding_file}")
    else:
        logging.error("Failed to get a valid response for comparative analysis.")
        sys.exit(1)

    logging.info("LLM Mining script finished successfully.")

if __name__ == "__main__":
    main() 