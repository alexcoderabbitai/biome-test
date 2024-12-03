import os
import subprocess
import sys
import anthropic
from dotenv import load_dotenv
import patch  # Import the patch module
from unidiff import PatchSet
import git
from utls import parse_arguments, get_file_content, read_error_pattern, write_file_content, extract_project_name, create_new_branch, commit_changes
import difflib
import re
import tempfile
from utls import  apply_patch_to_content, process_single_file
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables
api_key = os.getenv('ANTHROPIC_API_KEY')
model = os.getenv('ANTHROPIC_MODEL')

# Use these variables when initializing your Anthropic client
client = anthropic.Client(api_key=api_key)


def handle_multi_file_errors(file_paths, error_number, commit_to_branch):
    print(f"Handling multi-file errors for files: {file_paths} with error number: {error_number}")
    
    # Concatenate all file contents into a single context for Anthropic model
    concatenated_file_content = ""
    for file_path in file_paths:
        concatenated_file_content += f"<--[file-{file_path}]-->\n"
        file_content = get_file_content(file_path)
        numbered_file_content = "\n".join([f"{i+1}: {line}" for i, line in enumerate(file_content.splitlines())])
        concatenated_file_content += numbered_file_content + f"\n<!--[file-{file_path}]-->\n\n"

    # Read error pattern
    error_pattern = read_error_pattern("error-bot/Error-patterns", error_number)
    if not error_pattern:
        print(f"Error pattern {error_number} not found. Skipping multi-file operation.")
        return

    # Updated System message
    # system_message = (
    #     "You are an expert programmer with deep knowledge of software architecture, design patterns, and complex system interactions. "
    #     "Your task is to introduce subtle, intricate errors across multiple files of a project. These errors should be challenging to detect "
    #     "and may involve interactions between different components. Create a sandbox environment that tests advanced error patterns. "
    #     "Modify the given code files based on the specified error, ensuring the changes are non-obvious and could potentially pass thorough code reviews. "
    #     "The errors should be logically consistent with the existing code and appear as plausible implementations. "
    #     "Provide changes for each file as a unified diff format. Include proper headers with filenames and hunk headers in each diff. "
    #     "Do not add comments that reveal the error is intentional. The modifications should maintain the overall structure and style of the original code."
    # )
    system_message = (
        "You are an expert programmer tasked with introducing specific errors into multiple files of a project. This to create a sandbox environment for testing the error pattern."
        "Modify the given code files based on the specified error. Provide changes for each file as a unified diff format. "
        "Include proper headers with filenames and hunk headers in each diff. Do not add comments that reveal the error is intentional."
    )

    # Updated User message
    # user_message = (
    #     f"Here's the concatenated content of all files:\n\n{concatenated_file_content}\n\n"
    #     f"Please modify these files to subtly introduce the following error across multiple files: {error_pattern}\n"
    #     "Ensure that the error is complex, non-obvious, and could potentially pass thorough code reviews. "
    #     "The error should be logically consistent with the existing code and appear as a plausible implementation. "
    #     "Consider the following guidelines:\n"
    #     "1. Maintain the overall structure and style of the original code in each file.\n"
    #     "2. Introduce the error in a way that it affects the functionality but isn't immediately apparent.\n"
    #     "3. If possible, make the error dependent on interactions between multiple files or components.\n"
    #     "4. Avoid obvious syntax errors or changes that would cause immediate compilation/runtime errors.\n"
    #     "5. Consider introducing errors that might only manifest in certain environments, under specific loads, or in edge cases.\n"
    #     "6. Ensure that the error is distributed across multiple files in a logically consistent manner.\n\n"
    #     "Output the changes as unified diffs for each file. Include filenames in the diff headers. "
    #     "Enclose each diff within file-specific tags in the format <--[diff-{file_path}]--> and <!--[diff-{file_path}]-->."
    # )

    user_message = (
        f"Here's the concatenated content of all files:\n\n{concatenated_file_content}\n\n"
        f"Please modify these files to introduce the following error: {error_pattern}. "
        "Output the changes as unified diffs for each file. Include filenames in the diff headers. "
        "Enclose each diff within file-specific tags in the format <--[diff-{file_path}]--> and <!--[diff-{file_path}]-->."
    )


    # Get response from Anthropic model
    print("Sending request to Anthropic model...")
    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_message,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Extract the diffs from the response for each file
    full_response = response.content[0].text
    successfully_modified_files = []

    for file_path in file_paths:
        start_tag = f"<--[diff-{file_path}]-->"
        end_tag = f"<!--[diff-{file_path}]-->"
        start_index = full_response.find(start_tag)
        end_index = full_response.find(end_tag)

        if start_index != -1 and end_index != -1:
            start_index += len(start_tag)
            diff_content = full_response[start_index:end_index].strip()
            print(f"Received diff content from Anthropic for file: {file_path}")
        else:
            print(f"Error: Could not find diff tags in the response for file: {file_path}")
            continue

        # Determine output paths
        filename = os.path.basename(file_path)
        unique_identifier = filename.replace('.', '_')
        diff_output_path = os.path.join("error-bot", "code-test", f"diff_{unique_identifier}.patch")
        full_output_path = os.path.join("error-bot", "code-test", f"full_output_{unique_identifier}.txt")

        # Save the diff content
        write_file_content(diff_output_path, diff_content)
        print(f"Diff for error pattern {error_number} has been saved to {diff_output_path}")

        # Save the full output
        write_file_content(full_output_path, full_response)
        print(f"Full output has been saved to {full_output_path}")

        try:
            original_content = get_file_content(file_path)
            modified_content = apply_patch_to_content(original_content, diff_content, file_path)

            if modified_content == original_content:
                print(f"Error: Failed to apply the patch for file: {file_path}")
                continue
            else:
                print(f"Patch applied successfully for file: {file_path}")
        
            if commit_to_branch:
                code_output_path = os.path.abspath(file_path)
            else:
                code_output_path = os.path.join("error-bot", "code-test", os.path.basename(file_path))
            
            write_file_content(code_output_path, modified_content)
            print(f"Modified content has been saved to {code_output_path}")
            successfully_modified_files.append(file_path)
        except Exception as e:
            print(f"Error applying patch to {file_path}: {str(e)}")

    if commit_to_branch and successfully_modified_files:
        project_name = extract_project_name(file_paths[0])
        branch_name = create_new_branch(error_number, project_name)
        
        # Switch to the new branch
        subprocess.run(["git", "checkout", branch_name], capture_output=True, text=True)
        
        # Write modified content for each file
        for file_path in successfully_modified_files:
            write_file_content(file_path, modified_content)
            
        # Stage and commit all changes
        commit_message = f"Add error pattern {error_number:03d} to multiple files"
        commit_changes(file_paths[0], commit_message, branch_name)  # Using first file path as reference
        
    elif commit_to_branch:
        print("No files were successfully modified. Skipping branch creation and commit.")
    else:
        print(f"Changes have been made to {len(successfully_modified_files)} files, but not committed.")


def main(file_paths, error_number, commit_to_branch=False, multi_file=False):
    print(f"Starting main function with file_paths: {file_paths}, error_number: {error_number}, commit_to_branch: {commit_to_branch}, multi_file: {multi_file}")
    
    if multi_file:
        handle_multi_file_errors(file_paths, error_number, commit_to_branch)
    else:
        process_single_file(file_paths[0], error_number, commit_to_branch)

# Add test arguments here
# test_file_paths = ["project/api/llama_stack/apis/agents/event_logger.py", "project/api/llama_stack/apis/agents/agents.py"]
# test_error_number = "28"
# test_commit_to_branch = True  # Change this to a boolean False
# test_multi_file = True



if __name__ == "__main__":
    # Comment out the argument parsing
    args = parse_arguments()
    main(args.file_paths, args.error_number, args.commit, args.multi_file)

    # Use the test arguments instead
    # main(test_file_paths, test_error_number, test_commit_to_branch, test_multi_file)




