import argparse
import csv
import os
import subprocess
import sys
import patch  # Ensure this is the python-patch module from python-patch library
import tempfile
import os
import re
# import logging
import shutil
import anthropic
from dotenv import load_dotenv


# Load environment variables at the top of the file
load_dotenv()

# Initialize Anthropic client
client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))


def fix_hunk_headers_and_whitespace(diff_content):
    lines = diff_content.splitlines()
    fixed_lines = []
    current_hunk = []
    in_hunk = False

    for line in lines:
        if line.startswith('@@'):
            if in_hunk:
                # Process the previous hunk
                fixed_lines.extend(process_hunk(current_hunk))
                current_hunk = []
            in_hunk = True
            current_hunk.append(line)
        elif in_hunk:
            current_hunk.append(line)
        else:
            fixed_lines.append(line)

    if current_hunk:
        # Process the last hunk
        fixed_lines.extend(process_hunk(current_hunk))

    return '\n'.join(fixed_lines) + '\n'




def process_hunk(hunk):
    header = hunk[0]
    content = hunk[1:]
    
    # Parse the hunk header
    match = re.match(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', header)
    if match:
        old_start, old_count, new_start, new_count = map(int, match.groups())
        
        # Count actual lines
        actual_old_count = sum(1 for line in content if line.startswith(' ') or line.startswith('-'))
        actual_new_count = sum(1 for line in content if line.startswith(' ') or line.startswith('+'))
        
        # Adjust counts if necessary
        if actual_old_count < old_count:
            content.append(' ')
            actual_old_count += 1
            actual_new_count += 1
        
        # Update the header
        new_header = f'@@ -{old_start},{actual_old_count} +{new_start},{actual_new_count} @@'
        return [new_header] + content
    
    return hunk


def parse_arguments():
    parser = argparse.ArgumentParser(description="Modify code to introduce errors.")
    parser.add_argument("file_paths", nargs="+", help="Path(s) to the file(s) to be modified")
    parser.add_argument("error_number", type=int, help="Error number to introduce")
    parser.add_argument("--commit", action="store_true", help="Commit changes to a new branch")
    parser.add_argument("--multi-file", "-m", action="store_true", help="Apply error to multiple files with multi-file error patterns")
    
    args = parser.parse_args()
    return args

def get_file_content(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def read_error_pattern(file_path, error_number):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if i == error_number:
                return line.strip()
    return None


def write_file_content(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, 'w') as f:
        f.write(content)



def extract_project_name(file_path):
    parts = file_path.split(os.sep)
    if "project" in parts:
        project_index = parts.index("project")
        if project_index + 1 < len(parts):
            return parts[project_index + 1]
    return "unknown-project"

def create_new_branch(error_number, project_name):
    branch_name = f"error-{error_number:03d}-{project_name}"  # Changed to use 3-digit padding
    print(f"Creating new git branch: {branch_name}")
    # Create branch without checking out
    result = subprocess.run(["git", "branch", branch_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error creating branch: {result.stderr.strip()}")
        sys.exit(1)
    print(f"Successfully created branch: {branch_name}")
    return branch_name



def commit_changes(file_path, commit_message, branch_name):
    print(f"Committing changes to file: {file_path} on branch: {branch_name}")
    
    # Switch to the target branch
    subprocess.run(["git", "checkout", branch_name], capture_output=True, text=True)
    
    # Stage the changes
    subprocess.run(["git", "add", file_path], capture_output=True, text=True)
    
    # Commit the changes
    commit_result = subprocess.run(["git", "commit", "-m", commit_message, "--no-verify"], capture_output=True, text=True)
    if commit_result.returncode != 0:
        print(f"Error committing changes: {commit_result.stderr.strip()}")
        return False
    
    # Push the changes
    push_result = subprocess.run(["git", "push", "origin", branch_name], capture_output=True, text=True)
    if push_result.returncode != 0:
        print(f"Error pushing changes: {push_result.stderr.strip()}")
        return False
    
    # Switch back to the original branch (assuming it's main/master)
    subprocess.run(["git", "checkout", "main"], capture_output=True, text=True)
    
    print(f"Successfully committed and pushed changes to branch: {branch_name}")
    return True


def apply_patch_to_content(original_content, diff_content, file_path):
    # Add debug logging
    print("Original content first 20 lines:")
    print("\n".join(original_content.splitlines()[:20]))
    print("\nOriginal diff content:")
    print(diff_content)
    
    fixed_diff_content = fix_hunk_headers_and_whitespace(diff_content)

    file_name = os.path.basename(file_path)
    if isinstance(file_name, bytes):
        file_name = file_name.decode('utf-8')
    
    print("\nProcessed diff content:")
    print(fixed_diff_content)
    
    # Use simple filenames without a/ and b/ prefixes
    fixed_diff_content = re.sub(r'^--- .*$', f'--- {file_name}', fixed_diff_content, flags=re.MULTILINE)
    fixed_diff_content = re.sub(r'^\+\+\+ .*$', f'+++ {file_name}', fixed_diff_content, flags=re.MULTILINE)
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Write the original content to a file in the temporary directory
        original_file_path = os.path.join(temp_dir, file_name)
        with open(original_file_path, 'w') as original_file:
            original_file.write(original_content)

        # Write the fixed diff content to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.patch') as diff_file:
            diff_file.write(fixed_diff_content)
            diff_file_path = diff_file.name

        # Read the patch from the diff file
        pset = patch.fromfile(diff_file_path)
        
        if not pset:
            print("Failed to parse patch file")
            os.remove(diff_file_path)
            shutil.rmtree(temp_dir)
            return original_content

        # Apply the patch to the files in the temporary directory
        success = pset.apply(root=temp_dir)
        
        if not success:
            print("Failed to apply patch")
            os.remove(diff_file_path)
            shutil.rmtree(temp_dir)
            return original_content

        # Read the patched content from the file
        with open(original_file_path, 'r') as patched_file:
            patched_content = patched_file.read()

        return patched_content

    finally:
        # Clean up temporary files and directory
        if os.path.exists(diff_file_path):
            os.remove(diff_file_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)



def process_single_file(file_path, error_number, commit_to_branch):
    print(f"Processing single file: {file_path} with error number: {error_number}")
    
    # Extract project name
    project_name = extract_project_name(file_path)

    # Read error pattern
    error_pattern = read_error_pattern("error-bot/Error-patterns", error_number)
    if not error_pattern:
        print(f"Error pattern {error_number} not found. Skipping file: {file_path}")
        return

    # Read file content and add line numbers
    file_content = get_file_content(file_path)
    # Add line numbers to the file content
    numbered_file_content = "\n".join([f"{i+1}: {line}" for i, line in enumerate(file_content.splitlines())])

    # Updated System message
    # system_message = (
    #     "You are an expert programmer with deep knowledge of software architecture, design patterns, and common pitfalls in various programming languages. "
    #     "Your task is to introduce subtle, complex errors into code that are challenging to detect through standard code review processes. "
    #     "These errors should be logically consistent with the existing code and appear as plausible implementations. "
    #     "Modify the given code based on the specified error pattern, ensuring the changes are non-obvious and could potentially pass initial code reviews. "
    #     "Provide the changes as a unified diff format enclosed within <--[diff]--> and <!--[diff]--> tags. "
    #     "The diff should include proper headers with filenames and hunk headers. "
    #     "Do not add any comments that explicitly mention introducing an error. "
    #     "The modifications should maintain the overall structure and style of the original code."
    # )

    system_message = (
        "You are an expert programmer tasked with introducing specific errors into code. "
        "Modify the given code based on the specified error pattern. Provide the changes as a unified diff format "
        "enclosed within <--[diff]--> and <!--[diff]--> tags. The diff should include proper headers with filenames "
        "and hunk headers. Do not add any comments that explicitly mention introducing an error."
    )
      # User message
    user_message = (
        f"Here's the file content:\n\n{numbered_file_content}\n\n"
        f"Please modify this file to introduce the following error: {error_pattern}. "
        "Output the changes as a unified diff, including filenames in the diff headers. "
        "Do not add comments that reveal the error is intentional."
    )
    user_message += (
        "\n\nExample of the expected diff format:\n"
        "```\n"
        "--- original_file.py\n"
        "+++ modified_file.py\n"
        "@@ -1,4 +1,4 @@\n"
        "-def process_data(input_data):\n"
        "+def process_data(input_data, flag=False):\n"
        "```"
    )



    # Updated User message
    # user_message = (
    #     f"Here's the file content with line numbers:\n\n{numbered_file_content}\n\n"
    #     f"Please modify this file to subtly introduce the following error: {error_pattern}\n"
    #     "Output the changes as a unified diff, including filenames in the diff headers. "
    #     "Ensure that the error is complex, non-obvious, and could potentially pass initial code reviews. "
    #     "The error should be logically consistent with the existing code and appear as a plausible implementation. "
    #     "Consider the following guidelines:\n"
    #     "1. Maintain the overall structure and style of the original code.\n"
    #     "2. Introduce the error in a way that it affects the functionality but isn't immediately apparent.\n"
    #     "3. If possible, make the error dependent on specific conditions or edge cases.\n"
    #     "4. Avoid obvious syntax errors or changes that would cause immediate compilation/runtime errors.\n"
    #     "5. Consider introducing errors that might only manifest in certain environments or under specific loads.\n\n"
    #     "Output the changes as a unified diff with these exact headers:\n"
    #     "Do not add comments that reveal the error is intentional."
    # )
    # user_message += (
    #     "\n\nExample of the expected diff format:\n"
    #     "```\n"
    #     "--- original_file.py\n"
    #     "+++ modified_file.py\n"
    #     "@@ -1,4 +1,4 @@\n"
    #     "-def process_data(input_data):\n"
    #     "+def process_data(input_data, flag=False):\n"
    #     "```"
    # )
    # Get response from Anthropic model
    print("Sending request to Anthropic model...")
    
    # Create the Anthropic client with the API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    
    response = client.messages.create(
        model=os.getenv('ANTHROPIC_MODEL', "claude-3-5-sonnet-20240620"),
        max_tokens=8191,
        system=system_message,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    # Extract the diff from the response
    full_response = response.content[0].text
    start_tag = "<--[diff]-->"
    end_tag = "<!--[diff]-->"
    start_index = full_response.find(start_tag)
    end_index = full_response.find(end_tag)

    if start_index != -1 and end_index != -1:
        start_index += len(start_tag)
        diff_content = full_response[start_index:end_index].strip()
        print(f"Received diff content from Anthropic for file: {file_path}")
    else:
        print(f"Error: Could not find diff tags in the response for file: {file_path}")
        return
    
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
        modified_content = apply_patch_to_content(file_content, diff_content, file_path)

        if modified_content == file_content:
            print(f"Error: Failed to apply the patch for file: {file_path}")
            return
        else:
            print(f"Patch applied successfully for file: {file_path}")
    except Exception as e:
        print(f"Error applying diff for file {file_path}: {e}")
        return
    
    if commit_to_branch:
        # Create new branch
        branch_name = create_new_branch(error_number, project_name)
        
        # Switch to the new branch
        subprocess.run(["git", "checkout", branch_name], capture_output=True, text=True)
        
        # Write the modified content directly to the file
        write_file_content(file_path, modified_content)
        
        # Commit and push changes
        commit_message = f"Add error pattern {error_number:03d} in {os.path.basename(file_path)}"
        commit_changes(file_path, commit_message, branch_name)
    else:
        # Save to test directory when not committing
        code_output_path = os.path.join("error-bot", "code-test", os.path.basename(file_path))
        write_file_content(code_output_path, modified_content)
        print(f"Error pattern {error_number} has been added to {code_output_path} for testing")










