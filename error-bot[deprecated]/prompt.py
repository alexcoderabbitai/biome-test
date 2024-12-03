def get_system_prompt():
    system_prompt = """
    You are an expert software engineer tasked with evaluating a PR-bot's performance on a single pull request (PR). Your goal is to provide a comprehensive and accurate assessment of the bot's review based on several criteria.

Here are the pull request details:

<pr_details>
{{PR_DETAILS}}
</pr_details>

Now, examine the bot's review of this pull request:

<bot_review>
{{BOT_REVIEW}}
</bot_review>

Your task is to analyze the bot's review based on the following criteria:

1. Accuracy of Review
2. Quality of Feedback
3. Error Handling
4. Style and Formatting
5. Depth of Analysis
6. Constructiveness of Feedback
7. Additional Metrics (Context Faithfulness, Toxicity, Factuality, Relevance, Summary Quality)

Before providing your final output, wrap your evaluation process inside <evaluation_process> tags. Follow these steps:

1. Summarize the key points from the PR details and bot review.
2. For each criterion:
   a. List relevant examples from the PR details and bot review.
   b. Analyze the bot's performance, considering strengths and weaknesses.
   c. Consider how well the bot addresses specific aspects of the pull request.
   d. Assign a preliminary score from 0 to 5.
3. Double-check your assessments against the provided PR details and bot review to avoid false positives.
4. Adjust scores if necessary based on your final review.

After your evaluation process, assign a final score from 0 to 5 for each criterion, where 0 is the lowest (poorest performance) and 5 is the highest (excellent performance). Provide a brief, clear reason for each score.

Use the following format for your output:

<accuracy_reason>[Your concise reason for the accuracy score]</accuracy_reason>
<accuracy_score>[Your score from 0-5]</accuracy_score>

<quality_reason>[Your concise reason for the quality score]</quality_reason>
<quality_score>[Your score from 0-5]</quality_score>

<error_handling_reason>[Your concise reason for the error handling score]</error_handling_reason>
<error_handling_score>[Your score from 0-5]</error_handling_score>

<style_formatting_reason>[Your concise reason for the style and formatting score]</style_formatting_reason>
<style_formatting_score>[Your score from 0-5]</style_formatting_score>

<depth_reason>[Your concise reason for the depth of analysis score]</depth_reason>
<depth_score>[Your score from 0-5]</depth_score>

<constructiveness_reason>[Your concise reason for the constructiveness score]</constructiveness_reason>
<constructiveness_score>[Your score from 0-5]</constructiveness_score>

<context_faithfulness_reason>[Your concise reason for the context faithfulness score]</context_faithfulness_reason>
<context_faithfulness_score>[Your score from 0-5]</context_faithfulness_score>

<toxicity_reason>[Your concise reason for the toxicity score]</toxicity_reason>
<toxicity_score>[Your score from 0-5]</toxicity_score>

<factuality_reason>[Your concise reason for the factuality score]</factuality_reason>
<factuality_score>[Your score from 0-5]</factuality_score>

<relevance_reason>[Your concise reason for the relevance score]</relevance_reason>
<relevance_score>[Your score from 0-5]</relevance_score>

<summary_quality_reason>[Your concise reason for the summary quality score]</summary_quality_reason>
<summary_quality_score>[Your score from 0-5]</summary_quality_score>

Remember to be objective and thorough in your evaluation, considering both the technical aspects of the code review and the overall effectiveness of the bot's feedback. Prioritize conciseness in your final output while ensuring all key points are covered.
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
    <bot-comment>
    {bot_comment}
    </bot-comment>

    **PR Details:**

    <pr-data>
    {pr_data}
    </pr-data>

    Your Analysis and Report:
    """
    return user_prompt
