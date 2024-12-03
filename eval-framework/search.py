"""
GitHub Pull Request Search and Processing Tool
===========================================

This script provides functionality to search and process GitHub Pull Requests (PRs) based on
specified criteria defined in a JSON configuration file.

Features:
- Load PR search criteria from a JSON configuration file
- Search GitHub PRs using the GitHub API
- Handle rate limiting automatically
- Save search results to JSON files
- Process multiple programming languages and PR statuses
- Comprehensive logging

Required Environment Variables:
- GITHUB_TOKEN: GitHub Personal Access Token with repo access

Configuration File Format (PRs.json):
{
    "languages": [
        {
            "name": "Python",
            "statuses": {
                "approved": {
                    "query": "language:Python is:pr is:merged label:approved",
                    "pr_count": 50
                },
                "rejected": {
                    "query": "language:Python is:pr is:closed -is:merged",
                    "pr_count": 50
                }
            }
        }
    ]
}

Usage:
1. Create a .env file with your GitHub token:
   GITHUB_TOKEN=your_token_here

2. Prepare your PRs.json configuration file with search criteria

3. Run the script:
   python search.py

The script will:
- Process each language and status combination
- Save partial results to 'pr_results_partial.json'
- Save final results to 'pr_results_final.json'
"""

import json
import logging
from typing import List, Dict, Any
from github import Github, RateLimitExceededException
from dotenv import load_dotenv
import os
import time
import warnings

# Suppress urllib3 warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Github client
g = Github(os.getenv('GITHUB_TOKEN'))

def load_pr_data(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logging.info(f"Successfully loaded PR data from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
    return {}

def search_prs(query: str, max_results: int = 100) -> List[str]:
    urls = []
    try:
        logging.info(f"Searching for PRs with query: '{query}'")
        results = g.search_issues(query=query)
        for pr in results[:max_results]:
            urls.append(pr.html_url)
            if len(urls) >= max_results:
                break
            
            # Check rate limit after each request
            if g.get_rate_limit().search.remaining == 0:
                reset_time = g.get_rate_limit().search.reset.timestamp()
                sleep_time = reset_time - time.time() + 1  # Add 1 second buffer
                if sleep_time > 0:
                    logging.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
        
        logging.info(f"Found {len(urls)} PRs matching the search query.")
        return urls
    except RateLimitExceededException:
        logging.error("Rate limit exceeded. Please try again later.")
        return urls
    except Exception as e:
        logging.error(f"Error searching PRs: {str(e)}")
        return urls

def save_results_to_json(results: List[Dict[str, Any]], file_path: str):
    try:
        with open(file_path, 'w') as file:
            json.dump(results, file, indent=2)
        logging.info(f"Results saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results to {file_path}: {str(e)}")

def process_languages(pr_data: Dict[str, Any]) -> None:
    all_results = []
    for language in pr_data.get('languages', []):
        language_name = language['name']
        logging.info(f"Processing language: {language_name}")
        for status, status_data in language['statuses'].items():
            query = status_data['query']
            pr_count = status_data['pr_count']
            logging.info(f"Searching for {pr_count} {status} PRs in {language_name}")
            pr_urls = search_prs(query=query, max_results=pr_count)
            
            for url in pr_urls:
                result = {
                    'language': language_name,
                    'status': status,
                    'url': url
                }
                all_results.append(result)
            
            logging.info(f"Processed {len(pr_urls)} PRs for {language_name} - {status}")
            
            # Save results after each search
            save_results_to_json(all_results, "pr_results_partial.json")
            
            # Add a small delay between requests to avoid hitting rate limits
            time.sleep(2)
    
    # Save final results
    save_results_to_json(all_results, "pr_results_final.json")
    logging.info(f"Total PRs processed and saved: {len(all_results)}")

if __name__ == "__main__":
    logging.info("Starting PR search process")
    pr_data = load_pr_data("PRs.json")
    
    if pr_data:
        logging.info("Processing languages and searching for PRs")
        process_languages(pr_data)
    else:
        logging.error("Failed to load PR data. Exiting.")

    logging.info("PR search process completed")

# Example usage:
# pr_data = load_pr_data("eval-framework/PRs.json")
# query = "language:Python comments:7..20 commenter:app/coderabbitai is:pr"
# results = search_prs(query=query, pr_data=pr_data, max_results=50)
# save_results_to_json(results, "pr_results.json")
