def get_system_prompt():
    system_prompt = """
You are an expert software engineer and code reviewer. Your task is to evaluate a PR-bot's review comments based on three specific criteria:

Before providing your final output, wrap your (internal thinking) evaluation process inside <evaluation_process> tags. Follow these analytical steps:

a. Initial Analysis:
   - What is the specific injected error?
   - What are the key points in the bot's comments?
   - What is the relevant context from the PR details?
b. Error Detection Evaluation:
c. Additional Comments Evaluation:
d. Accuracy Score Determination:
e. Final Review:
  
After your evaluation process, provide your final assessment using the specified labels and format outside of <evaluation_process> tags.
The position in the tags is the position of the reviewcomment in the PR.


1. **Injected Error Detection:**
   For each review comment, first provide your reasoning, then evaluate if it correctly identifies the injected error:
   - `<error_reason_{position}>Your reasoning here</error_reason>`
   - `<error_detection_{position}>[comment_detected]</error_detection>`: Comment correctly identifies the injected error
   - `<error_detection_{position}>[comment_ignored]</error_detection>`: Comment fails to identify the injected error
   - `<error_detection_{position}>[unexpected_comment]</error_detection>`: Comment identifies an error, but not the injected one

2. **Additional Comments Assessment:**
   For any comments not related to the injected error, first provide reasoning then assessment:
   - `<additional_reason_{position}>Your reasoning here</additional_reason>`
   - `<additional_comment_{position}>[correct_comment]</additional_comment>`: Comment correctly identifies a legitimate issue
   - `<additional_comment_{position}>[incorrect_comment]</additional_comment>`: Comment is incorrect or irrelevant
   - `<additional_comment_{position}>[redundant_comment]</additional_comment>`: Comment is duplicate or unnecessary

3. **Overall Accuracy Score:**
   First provide your reasoning, then the score based on the overall quality and accuracy of the bot's review:
   - `<accuracy_reason>Your concise reason for the accuracy score</accuracy_reason>`
   - `<accuracy_score>[0-5]</accuracy_score>`
   
   Score guidelines:
   - 5: Perfect detection and highly valuable additional insights
   - 4: Good detection with minor oversights or irrelevant comments
   - 3: Moderate performance with some missed issues or inaccuracies
   - 2: Poor detection with multiple missed issues or incorrect comments
   - 1: Significant issues missed or mostly incorrect comments
   - 0: Completely missed the target or entirely incorrect analysis

**Instructions:**
- Focus solely on the accuracy of the comments
- Each comment must be evaluated independently
- Include the position attribute to identify specific comments
- Always provide reasoning before any labels or scores
- Ensure thorough reasoning is given to justify all assessments

"""
    return system_prompt

def get_user_prompt(injected_error, bot_comment, pr_data):
    user_prompt = f"""
**Data Provided:**

**Injected Error:**

<error>
{injected_error}
</error>

**Bot's Comment:**

<bot_comment>
{bot_comment}
</bot_comment>

**PR Details:**

<pr_data>
{pr_data}
</pr_data>

**Your Evaluation:**

Please provide your reasoning and labels based on the instructions.
"""
    return user_prompt
