#!/usr/bin/env python3

from Bio import Entrez
import xml.etree.ElementTree as ET # Import ElementTree for XML parsing
import time 
import os 
import argparse 
import json 
from dotenv import load_dotenv 
import logging # Import logging module
import sys # Import sys for console logging handler
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Fetch email from environment variable
NCBI_EMAIL = os.getenv("EMAIL")
if not NCBI_EMAIL:
    # Keep initial setup prints for immediate feedback before logging starts
    print("Error: NCBI email address not found in environment variables.") 
    print("Please set the EMAIL variable in your .env file.")
    exit(1) # Exit if email is not set

Entrez.email = NCBI_EMAIL

# Optional: Provide your NCBI API key if you have one (increases rate limits)
# Entrez.api_key = os.getenv("NCBI_API_KEY") # Example if you have an API key

# --- Function to Fetch and Parse ---
def fetch_and_parse_pmc_fulltext(gene_name, max_results=10, output_dir="pmc_full_texts"):
    """
    Searches PMC Open Access for articles related to a gene, fetches their full XML,
    extracts metadata (PMCID, title, pub date) and full text, and saves them
    into a single JSON file for the gene.
    Returns the number of articles successfully processed and saved in the JSON.
    """

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Error creating directory {output_dir}: {e}")
            return 0 # Return 0 if directory creation fails

    # Construct the search query including the Open Access filter
    search_query = f'({gene_name}[Gene Name] OR {gene_name}[Title]) AND "open access"[filter]' 
    logging.info(f"Searching PMC Open Access for articles with '{gene_name}' in title (max {max_results})...")

    # --- Searching for Articles ---
    handle = None # Initialize handle
    pmcid_list = []
    try:
        # Use Entrez.esearch to find articles
        handle = Entrez.esearch(db="pmc", 
                               term=search_query, 
                               retmax=str(max_results),
                               sort='relevance') 
        search_results = Entrez.read(handle)
        pmcid_list = search_results["IdList"]
        found_count = len(pmcid_list)
        # Updated log message for clarity
        logging.info(f"Found {found_count} Open Access articles (up to {max_results}) for '{gene_name}', sorted by most recent publication date.") 

    except Exception as e:
        logging.error(f"An error occurred during ESearch for '{gene_name}': {e}")
        return 0 # Return 0 on search error

    finally:
        if handle:
            handle.close()

    # --- Fetching and Parsing Articles (if any were found) ---
    gene_articles_data = [] # List to hold article dictionaries for this gene
    if not pmcid_list:
        logging.info(f"No articles found matching the criteria for '{gene_name}'.")
    else:
        logging.info(f"--- Attempting to fetch and parse full text/metadata for {len(pmcid_list)} PMCIDs for '{gene_name}' ---")
        for i, pmcid in enumerate(pmcid_list):
            logging.info(f"  [{i+1}/{len(pmcid_list)}] Processing PMCID: {pmcid}...")
            fetch_handle = None
            # Initialize article data
            article_title = None
            publication_date_str = None
            full_body_text = None

            try:
                # 1. Fetch the article XML
                logging.debug(f"    Fetching XML for {pmcid}...") # Use debug for lower level info
                fetch_handle = Entrez.efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
                xml_content = fetch_handle.read()
                if isinstance(xml_content, bytes):
                    xml_string = xml_content.decode('utf-8', errors='ignore')
                else:
                    xml_string = xml_content
                logging.debug(f"    Successfully fetched XML for {pmcid}.")

                # 2. Parse the XML to extract metadata and full text
                logging.debug(f"    Parsing XML...")
                try:
                    root = ET.fromstring(xml_string)

                    # Extract Title
                    title_element = root.find('.//article-meta/title-group/article-title')
                    if title_element is not None:
                        article_title = "".join(title_element.itertext()).strip()

                    # Extract Publication Date
                    pub_date_element = root.find('.//article-meta/pub-date[@pub-type="epub"]')
                    if pub_date_element is None:
                         pub_date_element = root.find('.//article-meta/pub-date[@pub-type="ppub"]')
                    if pub_date_element is None:
                        pub_date_element = root.find('.//article-meta/pub-date')

                    if pub_date_element is not None:
                        year = pub_date_element.findtext('year')
                        month = pub_date_element.findtext('month')
                        day = pub_date_element.findtext('day')
                        date_parts = [part for part in [year, month, day] if part]
                        publication_date_str = "-".join(date_parts) if date_parts else None

                    # Extract Full Text from Body
                    body_element = root.find('.//body')
                    if body_element is not None:
                        paragraphs = [para.strip() for para in body_element.itertext() if para and para.strip()]
                        full_body_text = "\n\n".join(paragraphs)

                    # Append extracted data
                    if full_body_text or article_title:
                        article_data = {
                            "pmcid": pmcid,
                            "title": article_title,
                            "publish_date": publication_date_str,
                            "full_text": full_body_text
                        }
                        gene_articles_data.append(article_data)
                        # Log success including the publication date
                        logging.debug(f"    Successfully parsed data for PMCID {pmcid} (Pub Date: {publication_date_str}).") 
                    else:
                        logging.warning(f"    Could not extract sufficient data (title/text) for {pmcid}.")

                except ET.ParseError as e:
                    logging.error(f"    Error parsing the XML for {pmcid}: {e}")
                except Exception as e:
                    logging.error(f"    An unexpected error occurred during XML parsing for {pmcid}: {e}")

            except Exception as e:
                logging.error(f"  An error occurred during fetching or processing {pmcid} for '{gene_name}': {e}")

            finally:
                if fetch_handle:
                    fetch_handle.close()
                time.sleep(0.5)

    # --- Save collected data to JSON ---
    processed_count = len(gene_articles_data)
    if processed_count > 0:
        output_file_path = os.path.join(output_dir, f"{gene_name}_pmc_articles.json")
        logging.info(f"Saving {processed_count} processed articles for '{gene_name}' to '{output_file_path}'...")
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(gene_articles_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved JSON file for '{gene_name}'.")
        except IOError as e:
            logging.error(f"Error writing JSON file for '{gene_name}': {e}")
            processed_count = 0 # Failed to save, count as 0
        except TypeError as e:
             logging.error(f"Error serializing data to JSON for '{gene_name}': {e}")
             processed_count = 0 # Failed to save, count as 0
    else:
        logging.info(f"No articles processed or extracted for '{gene_name}'. No JSON file saved.")

    return processed_count

# --- Helper function to read gene list ---
def read_gene_list(file_path):
    """Read gene names list from a text file, one gene per line"""
    genes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                gene = line.strip()
                if gene:
                    genes.append(gene)
        if not genes:
            logging.warning(f"No gene names found in file: {file_path}")
        return genes
    except FileNotFoundError:
        logging.error(f"Gene list file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading gene file {file_path}: {e}")
        return None

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Logging Setup --- 
    log_dir = 'logs' # Define log directory name
    
    # Create timestamp for log file name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_filename = f"pmc_fetcher_{timestamp}.log" # Construct filename with timestamp
    log_file = os.path.join(log_dir, log_filename) # Define full log file path

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, # Set default logging level (can be changed via args later)
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file), # Log to timestamped file inside 'logs' directory
            logging.StreamHandler(sys.stdout) # Log to console
        ]
    )
    logging.info(f"Starting PMC Full Text Fetcher script... Logging to {log_file}") # Log the filename
    # --- End Logging Setup ---

    parser = argparse.ArgumentParser(description="Fetch metadata and full text for Open Access PMC articles related to genes and save as JSON.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--gene', '-g',
                           help='Single gene name to search for.')
    input_group.add_argument('--list', '-l',
                           help='Path to text file with gene names list (one per line).')

    parser.add_argument("-n", "--max_results",
                        type=int,
                        default=10,
                        help="Maximum number of articles to fetch per gene (default: 10).")
    parser.add_argument("-o", "--output_dir",
                        default="pmc_full_texts",
                        help="Directory to save the JSON files (default: pmc_full_texts).")

    args = parser.parse_args()
    logging.info(f"Arguments received: {args}")

    total_processed_articles = 0
    processed_gene_count = 0

    if args.gene:
        logging.info(f"Processing single gene: {args.gene}")
        processed_count = fetch_and_parse_pmc_fulltext(args.gene, args.max_results, args.output_dir)
        total_processed_articles = processed_count
        if processed_count > 0:
            processed_gene_count = 1
        logging.info(f"Finished processing '{args.gene}'. Saved data for {processed_count} articles to JSON.")

    elif args.list:
        logging.info(f"Reading gene list from: {args.list}")
        genes = read_gene_list(args.list)

        if genes:
            num_genes = len(genes)
            logging.info(f"Found {num_genes} genes in the file. Processing sequentially...")

            for i, gene in enumerate(genes, 1):
                logging.info(f"[{i}/{num_genes}] Processing gene: {gene}")
                processed_count = fetch_and_parse_pmc_fulltext(gene, args.max_results, args.output_dir)
                total_processed_articles += processed_count
                if processed_count > 0:
                    processed_gene_count += 1
                logging.info(f"-- Finished processing '{gene}'. Saved data for {processed_count} articles to JSON for this gene. --")

            logging.info(f"Completed processing {num_genes} genes from the list.")
        else:
            logging.error("Could not read genes from the list file. Exiting.")
            sys.exit(1)

    # Final summary
    logging.info(f"--- Overall Summary ---")
    logging.info(f"Successfully processed {processed_gene_count} gene(s).")
    logging.info(f"Total articles saved across all JSON files: {total_processed_articles}")
    logging.info(f"Output directory: {os.path.abspath(args.output_dir)}")

    logging.info("--- Script Finished ---") 