"""
GitHub Pull Request Analysis Pipeline
===================================

This script processes and analyzes GitHub Pull Requests (PRs) at scale, collecting comprehensive
data about PR contents, reviews, comments, and code changes.

Requirements:
------------
- GitHub API token in .env file (GITHUB_TOKEN)
- Input file: 'pr_results_final.json' containing list of PRs to analyze
    Format: [{"url": "github_pr_url", "language": "programming_language"}, ...]

Usage:
------
1. Set up environment:
   - Create .env file with GITHUB_TOKEN=your_github_token
   - Install requirements: pip install -r requirements.txt

2. Prepare input file:
   - Ensure pr_results_final.json exists with proper format

3. Run the script:
   python pipeline.py

Output:
-------
- Creates 'language_results' directory
- Generates JSON files for each programming language: pr_data_{language}.json
- Each file contains detailed PR data including:
  * Basic PR information (title, state, timestamps)
  * Comments and review comments
  * Code changes and diffs
  * Commit details
  * File changes and hunks

Error Handling:
--------------
- Implements retry mechanisms for API failures
- Handles GitHub API rate limiting
- Saves intermediate results every 10 PRs
- Parallel processing with thread pooling (5 workers)
"""

import os
import sys
import json
import requests
from github import Github, RateLimitExceededException
from dotenv import load_dotenv
import re
from urllib.parse import urlparse
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings

# Suppress InsecureRequestWarning
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable the warning
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

# Load environment variables from .env file
load_dotenv(dotenv_path='.env')

# Load the GitHub token from environment variable
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    logging.error("GITHUB_TOKEN not found in .env file or environment variables.")
    sys.exit(1)

# Initialize GitHub client with retry mechanism
def create_github_client():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    
    return Github(GITHUB_TOKEN, per_page=100, retry=5, timeout=60, verify=False)

g = create_github_client()

def fetch_with_rate_limit(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitExceededException as e:
            reset_time = int(e.headers.get('X-RateLimit-Reset', 0))
            sleep_time = max(reset_time - time.time(), 0) + 60  # Add 60 seconds buffer
            logging.warning(f"Rate limit exceeded. Sleeping for {sleep_time/60:.2f} minutes.")
            time.sleep(sleep_time)
        except Exception as e:
            logging.error(f"Error in fetch_with_rate_limit: {str(e)}")
            sleep_time = 300  # Sleep for 5 minutes on other exceptions
            logging.warning(f"Unexpected error. Sleeping for {sleep_time/60:.2f} minutes.")
            time.sleep(sleep_time)

def fetch_pr_data(repo_full_name, pr_number):
    # Fetch detailed data for a specific pull request
    logging.info(f"Fetching data for PR #{pr_number} in repository {repo_full_name}")
    
    try:
        # Get repository and pull request objects from GitHub API
        repo = fetch_with_rate_limit(g.get_repo, repo_full_name)
        pr = fetch_with_rate_limit(repo.get_pull, pr_number)
    except Exception as e:
        logging.error(f"Error fetching PR #{pr_number} from {repo_full_name}: {str(e)}")
        time.sleep(300)  # Wait for 5 minutes before retrying
        try:
            repo = fetch_with_rate_limit(g.get_repo, repo_full_name)
            pr = fetch_with_rate_limit(repo.get_pull, pr_number)
        except Exception as e:
            logging.error(f"Failed to fetch PR #{pr_number} after retry: {str(e)}")
            raise
    
    # Collect detailed PR data
    pr_data = {
        'number': pr.number,
        'title': pr.title,
        'user': pr.user.login if pr.user else None,
        'state': pr.state,
        'created_at': pr.created_at.isoformat() if pr.created_at else None,
        'closed_at': pr.closed_at.isoformat() if pr.closed_at else None,
        'merged_at': pr.merged_at.isoformat() if pr.merged_at else None,
        'merge_commit_sha': pr.merge_commit_sha,
        'body': pr.body,
        'labels': [label.name for label in pr.get_labels()],
        'commits': pr.get_commits().totalCount,
        'additions': pr.additions,
        'deletions': pr.deletions,
        'changed_files': pr.changed_files,
        'comments': [],
        'review_comments': [],
        'reviews': [],
        'requested_reviewers': [],
        'hunks': [],
        'commits_data': [],
        'file_changes': []
    }

    # Fetch commit details for the PR
    for commit in pr.get_commits():
        pr_data['commits_data'].append({
            'sha': commit.sha,
            'author': commit.commit.author.name,
            'message': commit.commit.message,
            'date': commit.commit.author.date.isoformat(),
            'files_changed': [{
                'filename': f.filename,
                'additions': f.additions,
                'deletions': f.deletions,
                'changes': f.changes,
                'status': f.status
            } for f in commit.files]
        })

    # Fetch comments on the PR
    for comment in pr.get_issue_comments():
        pr_data['comments'].append({
            'user': comment.user.login,
            'created_at': comment.created_at.isoformat(),
            'body': comment.body
        })

    # Fetch review comments on the PR
    for review_comment in pr.get_review_comments():
        # Calculate the from and to line numbers for the comment
        from_line = review_comment.original_position
        to_line = review_comment.position if review_comment.position else from_line
        pr_data['review_comments'].append({
            'user': review_comment.user.login,
            'created_at': review_comment.created_at.isoformat(),
            'body': review_comment.body,
            'path': review_comment.path,
            'position': review_comment.position,
            'line_range': f"Comment on lines +{from_line} to +{to_line}"
        })

    # Fetch reviews on the PR
    for review in pr.get_reviews():
        pr_data['reviews'].append({
            'user': review.user.login,
            'state': review.state,
            'submitted_at': review.submitted_at.isoformat(),
            'body': review.body
        })

    # Fetch the diff using the GitHub API directly
    url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3.diff'
    }

    #print(f"Fetching diff for PR #{pr_number} from {repo_full_name}")
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch diff for PR #{pr_number}: {response.content}")
        return pr_data

    diff = response.text
    
    # Parse the diff to extract hunk information
    file_pattern = re.compile(r'diff --git a/(.*) b/(.*)')
    hunk_pattern = re.compile(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@')
    
    current_file = None
    hunk_id = 0
    
    for line in diff.split('\n'):
        file_match = file_pattern.match(line)
        if file_match:
            current_file = file_match.group(2)
            pr_data['file_changes'].append({'file': current_file, 'hunks': []})
           # print(f"Processing file: {current_file}")
            continue
        
        hunk_match = hunk_pattern.match(line)
        if hunk_match:
            hunk_id += 1
            old_start, old_count, new_start, new_count = map(int, hunk_match.groups())
        #   print(f"Found hunk #{hunk_id} in file {current_file}: old_start={old_start}, old_count={old_count}, new_start={new_start}, new_count={new_count}")
            hunk_data = {
                'id': hunk_id,
                'old_start': old_start,
                'old_count': old_count,
                'new_start': new_start,
                'new_count': new_count,
                'content': ""
            }
            pr_data['file_changes'][-1]['hunks'].append(hunk_data)
            continue
        
        if pr_data['file_changes'] and pr_data['file_changes'][-1]['hunks'] and line.startswith(('+', '-', ' ')):
            pr_data['file_changes'][-1]['hunks'][-1]['content'] += line + '\n'
    
    print(f"PR #{pr_number}")
    return pr_data

def parse_pr_url(url):
    # Parse the GitHub PR URL to extract the repository name and PR number
  #  print(f"Parsing PR URL: {url}")
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) < 4 or path_parts[2] != 'pull':
        raise ValueError("Invalid GitHub PR URL")
    repo_full_name = f"{path_parts[0]}/{path_parts[1]}"
    pr_number = int(path_parts[3])
 #   print(f"Parsed URL into repo: {repo_full_name}, PR number: {pr_number}")
    return repo_full_name, pr_number

def process_pr(pr_info):
    max_retries = 3
    retry_delay = 600  # 10 minutes

    for attempt in range(max_retries):
        try:
            repo_full_name, pr_number = parse_pr_url(pr_info['url'])
            pr_data = fetch_pr_data(repo_full_name, pr_number)
            pr_data['language'] = pr_info['language']
            return pr_data
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Error processing PR {pr_info['url']} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logging.info(f"Retrying in {retry_delay/60:.2f} minutes...")
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to process PR {pr_info['url']} after {max_retries} attempts: {str(e)}")
                return None

def save_language_results(language_groups):
    os.makedirs('language_results', exist_ok=True)
    for language, data in language_groups.items():
        output_file = f'language_results/pr_data_{language.lower().replace(" ", "_")}.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        logging.info(f"Saved {len(data)} PRs for language '{language}' to {output_file}")

def run_pipeline():
    # Read PR data from pr_results_final.json
    with open('pr_results_final.json', 'r') as f:
        pr_results = json.load(f)

    logging.info(f"Processing {len(pr_results)} PRs")

    # Process PRs in parallel
    language_groups = {}
    batch_size = 10
    processed_count = 0

    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced max_workers to 5
        future_to_pr = {executor.submit(process_pr, pr_info): pr_info for pr_info in pr_results}
        
        for future in tqdm(as_completed(future_to_pr), total=len(pr_results), desc="Processing PRs"):
            try:
                pr_data = future.result(timeout=1800)  # Set a timeout of 30 minutes
                if pr_data:
                    language = pr_data['language']
                    if language not in language_groups:
                        language_groups[language] = []
                    language_groups[language].append(pr_data)
                    processed_count += 1
            except Exception as e:
                logging.error(f"Error processing PR: {str(e)}")

            # Save intermediate results every batch_size PRs
            if processed_count % batch_size == 0:
                save_language_results(language_groups)
                logging.info(f"Processed {processed_count} PRs so far")

    # Save final results
    save_language_results(language_groups)

    logging.info("Finished processing all PRs")

if __name__ == '__main__':
    logging.info("Starting pipeline to process PRs from pr_results_final.json")
    run_pipeline()