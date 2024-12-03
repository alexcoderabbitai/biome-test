import json
import csv
import anthropic
import logging
from prompt_walkthrough import get_system_prompt, get_user_prompt
import os
from dotenv import load_dotenv
import pandas as pd
import sys
import re


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

def extract_walkthrough_comment(pr_data):
    """Extract the walkthrough comment from PR data."""
    walkthrough_comments = {}
    for pr in pr_data:
        walkthrough_comment = ''
        if 'comments' in pr:
            for comment in pr['comments']:
                comment_body = comment.get('body', '')
                if '<!-- walkthrough_start -->' in comment_body:
                    # Extract content between walkthrough markers
                    start = comment_body.find('<!-- walkthrough_start -->') + len('<!-- walkthrough_start -->')
                    end = comment_body.find('<!-- walkthrough_end -->')
                    if end > start:
                        walkthrough_comment = comment_body[start:end].strip()
                    break
        walkthrough_comments[pr['id']] = walkthrough_comment
    return walkthrough_comments

def extract_scores_and_reasons(prompt_output):
    # Handle TextBlock object
    if hasattr(prompt_output, 'text'):
        prompt_output = prompt_output.text
    elif isinstance(prompt_output, list) and len(prompt_output) > 0 and hasattr(prompt_output[0], 'text'):
        prompt_output = prompt_output[0].text
    
    if not isinstance(prompt_output, str):
        prompt_output = str(prompt_output)
        
    scores = {
        'accuracy_score': None,
        'accuracy_reason': None,
        'helpfulness_score': None,
        'helpfulness_reason': None,
        'pass_fail': None,
        'pass_fail_reason': None
    }
    
    # Updated regex patterns to match the new system prompt criteria
    accuracy = re.search(r'<accuracy_justification>(.*?)</accuracy_justification>\s*<accuracy_score>(\d+)</accuracy_score>', 
                        prompt_output, re.DOTALL)
    if accuracy:
        scores['accuracy_reason'] = accuracy.group(1).strip()
        scores['accuracy_score'] = int(accuracy.group(2))

    helpfulness = re.search(r'<helpfulness_justification>(.*?)</helpfulness_justification>\s*<helpfulness_score>(\d+)</helpfulness_score>', 
                          prompt_output, re.DOTALL)
    if helpfulness:
        scores['helpfulness_reason'] = helpfulness.group(1).strip()
        scores['helpfulness_score'] = int(helpfulness.group(2))

    pass_fail = re.search(r'<pass_fail_justification>(.*?)</pass_fail_justification>\s*<pass_fail_verdict>(PASS|FAIL)</pass_fail_verdict>', 
                         prompt_output, re.DOTALL)
    if pass_fail:
        scores['pass_fail_reason'] = pass_fail.group(1).strip()
        scores['pass_fail'] = pass_fail.group(2)
    
    # Add logging to help debug extraction issues
    if not all([scores['accuracy_score'], scores['helpfulness_score'], scores['pass_fail']]):
        logging.warning(f"Failed to extract some scores. Current scores: {scores}")
        logging.debug(f"Raw output: {prompt_output}")
    
    return scores

def validate_score(score, min_val=1, max_val=5):
    try:
        score = int(score)
        if min_val <= score <= max_val:
            return score
        logging.warning(f"Score {score} outside valid range [{min_val}-{max_val}]")
    except (ValueError, TypeError) as e:
        logging.warning(f"Invalid score format: {score}")
    return None

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
            error_list = provided_errors.get(error_number, [])
            if error_list:
                bot_errors[pr['id']] = error_list[0]
            else:
                logging.warning(f"No error found for error_number: {error_number}")
                bot_errors[pr['id']] = None
    return bot_errors

def get_scores_from_model(pr_content, walkthrough_comment, injected_error):
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(pr_content, walkthrough_comment, injected_error)
    
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
        # Extract the text content from the message
        content = message.content[0].text if isinstance(message.content, list) else message.content
        logging.info(f"Model response: {content}")
        return content
    except Exception as e:
        logging.error(f"Failed to get scores from model: {e}")
        raise

def main():
    # Check if number argument is provided
    if len(sys.argv) != 2:
        print("Usage: python llm-as-a-judge-walkthrough-updated.py <number>")
        sys.exit(1)
    
    number = sys.argv[1]
    pr_file = f'eval_json/pr_tpye_{number}.json'
    pr_data = load_json_data(pr_file)
    excel_file = f'eval_json/pr_walkthrough_scores_updated_6_{number}.xlsx'
    
    # Add logging to show full path
    absolute_path = os.path.abspath(excel_file)
    logging.info(f"Will save Excel file to: {absolute_path}")
    
    # Create eval_json directory if it doesn't exist
    os.makedirs(os.path.dirname(excel_file), exist_ok=True)
    
    # Initialize all_rows list
    all_rows = []
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            all_rows = existing_df.to_dict('records')
            logging.info(f"Loaded {len(all_rows)} existing records from {excel_file}")
        except Exception as e:
            logging.warning(f"Could not load existing Excel file: {e}")

    # Add provided errors dictionary
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
    walkthrough_comments = extract_walkthrough_comment(pr_data)

    for pr in pr_data:
        pr_id = pr['id']
        pr_number = pr['number']
        head_branch = pr['head_branch']
        
        # Skip if this PR was already processed
        if any(row['PR Number'] == pr_number for row in all_rows):
            logging.info(f"Skipping PR {pr_number} as it was already processed")
            continue

        # Send the entire PR object instead of converting to string
        pr_content = pr
        # logging.info(f"PR_XXX content: {pr_content}")
        walkthrough_comment = walkthrough_comments.get(pr_id, '')
        # logging.info(f"PR_XXX Walkthrough comment: {walkthrough_comment}")

        injected_error = provided_errors.get(extract_error_number(pr['head_branch']), [])
        prompt_output = get_scores_from_model(pr_content, walkthrough_comment, injected_error)
        scores_and_reasons = extract_scores_and_reasons(prompt_output)

        # Verify all required scores are present
        required_fields = ['accuracy_score', 'helpfulness_score', 'pass_fail', 'pass_fail_reason']
        if not all(scores_and_reasons.get(field) for field in required_fields):
            logging.error(f"Missing required scores for PR {pr_number}. Scores: {scores_and_reasons}")
            continue

        # Create row with evaluation results
        row = {
            'PR Number': pr_number,
            'Head Branch': head_branch,
            'model_output': prompt_output,
            'accuracy_score': scores_and_reasons['accuracy_score'],
            'accuracy_reason': scores_and_reasons['accuracy_reason'],
            'helpfulness_score': scores_and_reasons['helpfulness_score'],
            'helpfulness_reason': scores_and_reasons['helpfulness_reason'],
            'pass_fail': scores_and_reasons['pass_fail'],
            'pass_fail_reason': scores_and_reasons['pass_fail_reason']
        }
        
        all_rows.append(row)
        # Modify the save part with error handling
        try:
            df = pd.DataFrame(all_rows)
            df.to_excel(excel_file, index=False)
            logging.info(f"Successfully saved {len(all_rows)} rows to {absolute_path}")
        except Exception as e:
            logging.error(f"Failed to save Excel file: {e}")
            raise
        logging.info(f"Processed and saved scores for PR {pr_id}")
        logging.info(f"================================================")

    logging.info(f"All processed scores and reasons have been written to {excel_file}")

if __name__ == '__main__':
    main() 