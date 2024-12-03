import json
import pandas as pd
import glob
import re
import csv

def get_experiment_details(pr_type):
    """Map PR type to experiment details"""
    type_mapping = {
        '1': {'experiment': 'test1', 'senior_reviewer': 'o1', 'anthropic_model': 'oldsonnet'},
        '2': {'experiment': 'test1-new', 'senior_reviewer': 'o1', 'anthropic_model': 'newsonnet'},
        '3': {'experiment': 'test2', 'senior_reviewer': 'newsonnet', 'anthropic_model': 'oldsonnet'},
        '4': {'experiment': 'test2-new', 'senior_reviewer': 'newsonnet', 'anthropic_model': 'newsonnet'}
    }
    return type_mapping.get(pr_type, {})

def clean_content(text):
    """Clean and escape content for CSV"""
    if not text:
        return ""
    return str(text).replace('\n', '\\n').replace('"', '""')

def extract_pr_data(pr_data):
    """Extract relevant data from a PR"""
    results = []
    for pr in pr_data:
        pr_url = f"github.com/coderabbitai/Golden-PR-Dataset/pull/{pr.get('number')}"
        pr_number = pr.get('number')
        pr_title = clean_content(pr.get('title'))
        
        # Extract walkthrough
        comments = pr.get('comments', [])
        pr_walkthrough = ''
        for comment in comments:
            body = comment.get('body', '')
            if '<!-- walkthrough_start -->' in body:
                pr_walkthrough = clean_content(body)
                break
        
        # Process file changes and their hunks
        file_changes = pr.get('file_changes', [])
        for file_change in file_changes:
            file_path = clean_content(file_change.get('file', ''))
            hunks = file_change.get('hunks', [])
            
            for hunk in hunks:
                hunk_position = f"{hunk.get('old_start')}-{hunk.get('new_start')}"
                hunk_content = clean_content(hunk.get('content', ''))
                
                # Find all review comments for this file
                hunk_comments = []
                for comment in comments:
                    if comment.get('path') == file_change.get('file'):
                        hunk_comments.append({
                            'path': clean_content(comment.get('path', '')),
                            'position': clean_content(comment.get('position', '')),
                            'body': clean_content(comment.get('body', ''))
                        })
                
                # Create row with base info
                row = {
                    'pr_url': pr_url,
                    'pr_number': pr_number,
                    'pr_title': pr_title,
                    'pr_walkthrough': pr_walkthrough,
                    'hunk_content': hunk_content,
                    'hunk_file_path': file_path,
                    'hunk_position': hunk_position
                }
                
                # Add all review comments for this file
                for i, comment in enumerate(hunk_comments, 1):
                    row[f'pr_review_comment_path_{i}'] = comment['path']
                    row[f'pr_review_comment_body_{i}'] = comment['body']
                    row[f'pr_review_comment_position_{i}'] = comment['position']
                
                results.append(row)
    
    return results

def process_json_files(json_files):
    """Process multiple JSON files and create a DataFrame"""
    all_data = []
    max_hunks = 0
    
    # First pass to determine maximum number of hunks
    for json_file in json_files:
        with open(json_file, 'r') as f:
            pr_data = json.load(f)
            extracted_data = extract_pr_data(pr_data)
            all_data.extend(extracted_data)
            
            # Find max hunk count by checking column names
            for row in extracted_data:
                hunk_columns = [col for col in row.keys() if col.startswith('pr_hunk_paths_')]
                max_hunks = max(max_hunks, len(hunk_columns))
    
    df = pd.DataFrame(all_data)
    
    # Define base columns
    base_columns = [
        'pr_url', 'pr_number', 'pr_title',
        'pr_walkthrough', 'hunk_content', 'hunk_file_path'
    ]
    
    # Add numbered hunk columns
    numbered_columns = []
    for i in range(1, max_hunks + 1):
        numbered_columns.extend([
            f'pr_hunk_paths_{i}',
            f'pr_hunk_bodies_{i}',
            f'pr_hunk_positions_{i}'
        ])
    
    # Ensure all required columns exist
    required_columns = base_columns + numbered_columns
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
    
    df = df[required_columns]
    return df

if __name__ == "__main__":
    json_files = glob.glob('error-bot/eval_json/pr_tpye_*.json')
    df = process_json_files(json_files)
    
    # Save to CSV with enhanced settings
    output_path = 'error-bot/eval_json/superset.csv'
    df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        doublequote=True,
        escapechar='\\',
        lineterminator='\n'
    )
    print(f"Data saved to {output_path}")
