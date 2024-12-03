import json
import csv
import anthropic
import logging
from prompt_label import get_system_prompt, get_user_prompt
import os
from dotenv import load_dotenv
import pandas as pd
import sys

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def load_json_data(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded JSON data from {file_path}")
    except Exception as e:
        logging.error(f"Failed to load JSON data from {file_path}: {e}")
        raise
    return data

def extract_error_number(head_branch):
    """Extract the error number from the head_branch."""
    try:
        return head_branch.split('-')[1]
    except IndexError:
        logging.error(f"Failed to extract error number from head_branch: {head_branch}")
        return None

def extract_bot_errors(pr_data, provided_errors):
    """Extract errors reported by the bot from PR data."""
    bot_errors = {}
    for pr in pr_data:
        error_number = extract_error_number(pr['head_branch'])
        if error_number:
            # Get the error list and check if it's not empty before accessing first element
            error_list = provided_errors.get(error_number, [])
            if error_list:  # Check if the list has any elements
                bot_errors[pr['id']] = error_list[0]
            else:
                logging.warning(f"No error found for error_number: {error_number}")
                bot_errors[pr['id']] = None  # or some default value
    return bot_errors

def extract_scores_and_reasons(prompt_output):
    if isinstance(prompt_output, list):
        prompt_output = prompt_output[0]
    if not isinstance(prompt_output, str):
        prompt_output = str(prompt_output)
        
    scores = {
        'positions': [],
        'error_detections': [],
        'error_reasons': [],
        'additional_reasons': [],
        'additional_comments': [],
        'accuracy_score': None,
        'accuracy_reason': None
    }
    
    import re
    
    # Find all error detections and their corresponding reasons
    error_reasons = re.finditer(r'<error_reason_(\d+)>(.*?)</error_reason_\1>', prompt_output, re.DOTALL)
    error_detections = re.finditer(r'<error_detection_(\d+)>(.*?)</error_detection_\1>', prompt_output, re.DOTALL)
    
    # Convert iterators to lists for easier pairing
    error_pairs = []
    for reason in error_reasons:
        position = reason.group(1)
        reason_text = reason.group(2).strip()
        
        # Find matching detection
        detection_value = None
        for detection in re.finditer(r'<error_detection_' + position + r'>(.*?)</error_detection_' + position + r'>', prompt_output, re.DOTALL):
            detection_text = detection.group(1).strip()
            if '[' in detection_text and ']' in detection_text:
                detection_value = detection_text[detection_text.find('[')+1:detection_text.find(']')]
            break
            
        scores['positions'].append(position)
        scores['error_reasons'].append(reason_text)
        scores['error_detections'].append(detection_value)
        scores['additional_reasons'].append(None)
        scores['additional_comments'].append(None)
    
    # Find all additional comment evaluations
    additional_reasons = re.finditer(r'<additional_reason_(\d+)>(.*?)</additional_reason_\1>', prompt_output, re.DOTALL)
    additional_comments = re.finditer(r'<additional_comment_(\d+)>(.*?)</additional_comment_\1>', prompt_output, re.DOTALL)
    
    for reason in additional_reasons:
        position = reason.group(1)
        reason_text = reason.group(2).strip()
        
        # Find matching comment
        comment_value = None
        for comment in re.finditer(r'<additional_comment_' + position + r'>(.*?)</additional_comment_' + position + r'>', prompt_output, re.DOTALL):
            comment_text = comment.group(1).strip()
            if '[' in comment_text and ']' in comment_text:
                comment_value = comment_text[comment_text.find('[')+1:comment_text.find(']')]
            break
            
        scores['positions'].append(position)
        scores['error_reasons'].append(None)
        scores['error_detections'].append(None)
        scores['additional_reasons'].append(reason_text)
        scores['additional_comments'].append(comment_value)
    
    # Extract accuracy score and reason
    accuracy_reason_start = prompt_output.find('<accuracy_reason>') + len('<accuracy_reason>')
    accuracy_reason_end = prompt_output.find('</accuracy_reason>')
    if accuracy_reason_start != -1 and accuracy_reason_end != -1:
        scores['accuracy_reason'] = prompt_output[accuracy_reason_start:accuracy_reason_end].strip()
        
    accuracy_score_start = prompt_output.find('<accuracy_score>') + len('<accuracy_score>')
    accuracy_score_end = prompt_output.find('</accuracy_score>')
    if accuracy_score_start != -1 and accuracy_score_end != -1:
        score_value = prompt_output[accuracy_score_start:accuracy_score_end].strip()
        if '[' in score_value and ']' in score_value:
            score_value = score_value[score_value.find('[')+1:score_value.find(']')]
        scores['accuracy_score'] = score_value
    
    return scores

def get_scores_from_model(injected_error, bot_comment, pr_details):
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(pr_details, bot_comment, injected_error)
    
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]
        )
        logging.info("Successfully received scores from model")
    except Exception as e:
        logging.error(f"Failed to get scores from model: {e}")
        raise
    return message.content

def extract_bot_comment(pr_data):
    """Extract the bot's comment from PR data."""
    bot_comments = {}
    for pr in pr_data:
        # Look for comment from coderabbitai[bot]
        for comment in pr['comments']:
                bot_comments[pr['id']] = comment['body']
                break
    return bot_comments

def main():
    # Check if number argument is provided
    if len(sys.argv) != 2:
        print("Usage: python llm-as-a-judge-label.py <number>")
        sys.exit(1)
    
    # Get number from command line argument
    number = sys.argv[1]
    
    # Use the provided number in file paths
    pr_file = f'eval_json/pr_tpye_{number}.json'
    pr_data = load_json_data(pr_file)
    excel_file = f'eval_json/pr_scores_{number}.xlsx'
    
    # Initialize all_rows list - try to load existing data if file exists
    all_rows = []
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            all_rows = existing_df.to_dict('records')
            logging.info(f"Loaded {len(all_rows)} existing records from {excel_file}")
        except Exception as e:
            logging.warning(f"Could not load existing Excel file: {e}")

    # Provided errors with error numbers
    provided_errors = {
        '004': ['Incorrect handling of rare edge cases'],
        '018': ['Subtle issues with caching mechanisms'],
        '005': ['Logical errors in complex conditional statements'],
        '037': ['Incorrect assumptions about API behavior'],
        '003': ['Race conditions in multi-threaded code'],
        '006': ['Incorrect implementations of design patterns'],
        '017': ['Improper handling of concurrent database transactions'],
        '030': ['Subtle issues with asynchronous code'],
        '040': ['Incorrect handling of null or undefined values in complex scenarios'],
        '019': ['Incorrect implementations of complex mathematical formulas'],
        '015': ['Subtle issues with character encoding and internationalization'],
        '043': ['Incorrect handling of long-running processes'],
        '020': ['Unintended consequences of code optimizations'],
        '027': ['Subtle issues with recursive algorithms'],
        '041': ['Improper implementations of caching invalidation strategies'],
        '038': ['Unintended consequences of code refactoring'],
        '016': ['Incorrect assumptions about network behavior'],
        '050': ['Unintended consequences of using third-party libraries'],
        '048': ['Subtle issues with database query optimization'],
        '013': ['Unintended side effects in pure functions'],
        '031': ['Incorrect handling of unicode characters in model responses'],
        '035': ['Improper handling of database connection pooling'],
        '011': ['Subtle security vulnerabilities (e.g. timing attacks)'],
        '026': ['Unintended consequences of lazy evaluation'],
        '029': ['Improper implementations of state machines'],
        '024': ['Subtle issues with floating-point comparisons'],
        '010': ['Improper error handling in distributed systems'],
        '047': ['Improper handling of network packet fragmentation'],
        '034': ['Incorrect implementations of custom hashing functions'],
        '046': ['Incorrect implementations of custom serialization/deserialization']
    }

    bot_errors = extract_bot_errors(pr_data, provided_errors)
    bot_comments = extract_bot_comment(pr_data)  # Add this line

    for pr in pr_data:
        pr_id = pr['id']
        pr_number = pr['number']
        head_branch = pr['head_branch']
        
        # Skip if this PR was already processed
        if any(row['PR Number'] == pr_number for row in all_rows):
            logging.info(f"Skipping PR {pr_number} as it was already processed")
            continue

        injected_error = provided_errors.get(extract_error_number(pr['head_branch']), [])
        bot_comment = bot_comments.get(pr_id, '')
        pr_details = pr

        prompt_output = get_scores_from_model(injected_error, bot_comment, pr_details)
        scores_and_reasons = extract_scores_and_reasons(prompt_output)

        # Create base row
        row = {
            'PR Number': pr_number,
            'Head Branch': head_branch,
            'model_output': prompt_output,
            'accuracy_score': scores_and_reasons['accuracy_score'],
            'accuracy_reason': scores_and_reasons['accuracy_reason'],
        }
        
        # Add numbered columns for each evaluation
        for i in range(len(scores_and_reasons['positions'])):
            row[f'position_{i+1}'] = scores_and_reasons['positions'][i]
            row[f'error_detection_{i+1}'] = scores_and_reasons['error_detections'][i]
            row[f'error_reason_{i+1}'] = scores_and_reasons['error_reasons'][i]
            row[f'additional_reason_{i+1}'] = scores_and_reasons['additional_reasons'][i]
            row[f'additional_comment_{i+1}'] = scores_and_reasons['additional_comments'][i]
        
        all_rows.append(row)
        df = pd.DataFrame(all_rows)
        df.to_excel(excel_file, index=False)
        logging.info(f"Processed and saved scores for PR {pr_id}")

        logging.info(f"================================================")
        # user_input = input("Press Enter to continue to the next PR or type 'exit' to stop: ")
        # if user_input.lower() == 'exit':
        #     break

    logging.info(f"All processed scores and reasons have been written to {excel_file}")

if __name__ == '__main__':
    main()
