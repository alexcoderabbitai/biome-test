"""
Word Counter for Pull Request Analysis
====================================

This script analyzes the verbosity of Pull Requests by counting words in various PR components
(body, comments, review comments, and reviews). It processes PR data from a JSON file and outputs
metrics to a CSV file.

Features:
- Counts words both including and excluding code blocks
- Handles markdown code blocks and HTML comments
- Supports incremental updates to the output CSV file
- Calculates separate metrics for PR body, comments, review comments, and reviews

Usage:
    1. Ensure your input JSON file contains PR data in the expected format
    2. Run the script directly:
        python word_counter.py
    
    Or import and use in another script:
        from word_counter import calculate_pr_verbosity
        calculate_pr_verbosity(json_data, output_file='pr_verbosity.csv')

Input JSON format expected:
{
    "number": int,
    "body": str,
    "comments": [{"body": str}, ...],
    "review_comments": [{"body": str}, ...],
    "reviews": [{"body": str}, ...]
}

Output CSV columns:
- PR_number: Pull Request number
- body_words: Word count in PR body (excluding code blocks)
- body_words_with_code: Word count in PR body (including code blocks)
- comments_words: Word count in PR comments (excluding code blocks)
- comments_words_with_code: Word count in PR comments (including code blocks)
- review_comments_words: Word count in review comments (excluding code blocks)
- review_comments_words_with_code: Word count in review comments (including code blocks)
- reviews_words: Word count in reviews (excluding code blocks)
- reviews_words_with_code: Word count in reviews (including code blocks)
- total_words: Total word count (excluding code blocks)
- total_words_with_code: Total word count (including code blocks)
- total_count: Same as total_words (for compatibility)
- total_count_with_code: Same as total_words_with_code (for compatibility)
"""

import json
import csv
import re
import os

def calculate_pr_verbosity(json_data, output_file='pr_verbosity.csv'):
    """
    Calculate verbosity metrics for each PR and save to CSV.
    
    Args:
        json_data (list): List of PR objects
        output_file (str): Output CSV filename
    """
    
    def count_words(text):
        if not text:
            return 0
        # Remove markdown code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Split and count remaining words
        return len(text.split())

    def count_words_with_code(text):
        if not text:
            return 0
        # Split and count words without removing code
        return len(text.split())

    results = []
    
    for pr in json_data:
        pr_metrics = {
            'PR_number': pr['number'],
            'body_words': count_words(pr.get('body', '')),
            'body_words_with_code': count_words_with_code(pr.get('body', '')),
            'comments_words': sum(count_words(c.get('body', '')) for c in pr.get('comments', [])),
            'comments_words_with_code': sum(count_words_with_code(c.get('body', '')) for c in pr.get('comments', [])),
            'review_comments_words': sum(count_words(c.get('body', '')) for c in pr.get('review_comments', [])),
            'review_comments_words_with_code': sum(count_words_with_code(c.get('body', '')) for c in pr.get('review_comments', [])),
            'reviews_words': sum(count_words(r.get('body', '')) for r in pr.get('reviews', [])),
            'reviews_words_with_code': sum(count_words_with_code(r.get('body', '')) for r in pr.get('reviews', []))
        }
        
        # Add total words
        pr_metrics['total_words'] = (pr_metrics['body_words'] + 
                                   pr_metrics['comments_words'] + 
                                   pr_metrics['review_comments_words'] + 
                                   pr_metrics['reviews_words'])
        
        pr_metrics['total_words_with_code'] = (pr_metrics['body_words_with_code'] + 
                                               pr_metrics['comments_words_with_code'] + 
                                               pr_metrics['review_comments_words_with_code'] + 
                                               pr_metrics['reviews_words_with_code'])
        
        # Add total count and total count with words
        pr_metrics['total_count'] = pr_metrics['total_words']
        pr_metrics['total_count_with_code'] = pr_metrics['total_words_with_code']
        
        results.append(pr_metrics)
    
    # Check if the file already exists and is not empty
    file_exists = os.path.isfile(output_file) and os.path.getsize(output_file) > 0
    
    # Write to CSV in append mode
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['PR_number', 'body_words', 'body_words_with_code', 
                                               'comments_words', 'comments_words_with_code', 
                                               'review_comments_words', 'review_comments_words_with_code', 
                                               'reviews_words', 'reviews_words_with_code', 
                                               'total_words', 'total_words_with_code',
                                               'total_count', 'total_count_with_code'])
        
        # Write header only if the file is new or empty
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(results)

    print(f"Results appended to {output_file}")

def main():
    # Hardcoded file paths
    input_file = 'eval_json/pr_tpye_8.json'
    output_file = 'eval_json/pr_verbosity.csv'
    
    with open(input_file, 'r') as f:
        json_data = json.load(f)
    
    calculate_pr_verbosity(json_data, output_file)

if __name__ == '__main__':
    main()