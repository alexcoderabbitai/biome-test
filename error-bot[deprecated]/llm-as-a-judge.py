import json
import csv
import anthropic
import logging
from prompt import get_system_prompt, get_user_prompt
import os
from dotenv import load_dotenv

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
    # Convert list to string if prompt_output is a list
    if isinstance(prompt_output, list):
        prompt_output = prompt_output[0]
    # If it's still not a string, convert it to one
    if not isinstance(prompt_output, str):
        prompt_output = str(prompt_output)
        
    scores = {}
    tags = [
        'accuracy_score', 'accuracy_reason',
        'quality_score', 'quality_reason',
        'error_handling_score', 'error_handling_reason',
        'consistency_score', 'consistency_reason',
        'thoroughness_score', 'thoroughness_reason',
        'constructiveness_score', 'constructiveness_reason',
        'context_faithfulness_score', 'context_faithfulness_reason',
        'toxicity_score', 'toxicity_reason',
        'factuality_score', 'factuality_reason',
        'relevance_score', 'relevance_reason',
        'summary_quality_score', 'summary_quality_reason'
    ]
    
    for tag in tags:
        start_tag = f'<{tag}>'
        end_tag = f'</{tag}>'
        start_index = prompt_output.find(start_tag) + len(start_tag)
        end_index = prompt_output.find(end_tag)
        scores[tag] = prompt_output[start_index:end_index].strip()
    
    return scores

def get_scores_from_model(injected_error, bot_comment, pr_details):
    system_prompt = get_system_prompt()
    user_prompt = get_user_prompt(injected_error, bot_comment, pr_details)
    
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
    pr_file = 'eval-framework/eval_json/pr_tpye_4.json'

    pr_data = load_json_data(pr_file)

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

    # Create and write header to CSV file immediately
    csv_file = 'eval-framework/eval_json/pr_scores_4.csv'  # Updated path
    fieldnames = ['PR Number', 'Head Branch'] + [
        'accuracy_score', 'accuracy_reason',
        'quality_score', 'quality_reason',
        'error_handling_score', 'error_handling_reason',
        'consistency_score', 'consistency_reason',
        'thoroughness_score', 'thoroughness_reason',
        'constructiveness_score', 'constructiveness_reason',
        'context_faithfulness_score', 'context_faithfulness_reason',
        'toxicity_score', 'toxicity_reason',
        'factuality_score', 'factuality_reason',
        'relevance_score', 'relevance_reason',
        'summary_quality_score', 'summary_quality_reason'
    ]
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for pr in pr_data:
        pr_id = pr['id']
        pr_number = pr['number']  # Get the PR number
        head_branch = pr['head_branch']  # Get the head branch name
        injected_error = provided_errors.get(extract_error_number(pr['head_branch']), [])
        bot_comment = bot_comments.get(pr_id, '')
        pr_details = pr

        prompt_output = get_scores_from_model(injected_error, bot_comment, pr_details)
        scores_and_reasons = extract_scores_and_reasons(prompt_output)

        # Update row dictionary to include PR number and head_branch
        row = {
            'PR Number': pr_number,
            'Head Branch': head_branch
        }
        row.update(scores_and_reasons)

        # Write this PR's data to CSV immediately
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(row)
        
        logging.info(f"Saved scores for PR {pr_id} to {csv_file}")

        logging.info(f"================================================")
        # # Ask for user input to continue
        # user_input = input("Press Enter to continue to the next PR or type 'exit' to stop: ")
        # if user_input.lower() == 'exit':
        #     break

    logging.info(f"All processed scores and reasons have been written to {csv_file}")

if __name__ == '__main__':
    main()
