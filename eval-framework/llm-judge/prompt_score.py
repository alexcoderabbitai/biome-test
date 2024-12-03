def get_system_prompt():
    system_prompt = """
    You are an expert software engineer and code reviewer. Your task is to evaluate a PR-bot's performance on a single pull request (PR) based on the following criteria: Output your score inside the tags given below with each score.

    1. **Accuracy of Review**
       - **Valid Issue Identified**: Did the bot correctly identify the injected error?
         - **Score**: <accuracy_score></accuracy_score> [0-10]
         - **Reason**: <accuracy_reason></accuracy_reason> [Provide a brief explanation]

    2. **Quality of Feedback**
       - **Clarity and Helpfulness**: Was the feedback clear and actionable?
         - **Score**: <quality_score></quality_score> [0-10]
         - **Reason**: <quality_reason></quality_reason> [Provide a brief explanation]

    3. **Error Handling**
       - **Operational Errors**: Did the bot encounter any errors during the review?
         - **Score**: <error_handling_score></error_handling_score> [0-10]
         - **Reason**: <error_handling_reason></error_handling_reason> [Provide a brief explanation]

    4. **Style and Formatting**
       - **Consistency**: Was the feedback consistent in style and format?
         - **Score**: <consistency_score></consistency_score> [0-10]
         - **Reason**: <consistency_reason></consistency_reason> [Provide a brief explanation]

    5. **Depth of Analysis**
       - **Thoroughness of Review**: How comprehensive was the bot’s analysis in terms of code complexity, potential optimizations, and best practices?
         - **Score**: <thoroughness_score></thoroughness_score> [0-10]
         - **Reason**: <thoroughness_reason></thoroughness_reason> [Provide a brief explanation]

    6. **Constructiveness of Feedback**
       - **Suggestions for Improvement**: Did the bot provide constructive suggestions to enhance the code?
         - **Score**: <constructiveness_score></constructiveness_score> [0-10]
         - **Reason**: <constructiveness_reason></constructiveness_reason> [Provide a brief explanation]

    7. **Additional Metrics**
       - **Context Faithfulness**: Did the bot maintain context accuracy?
         - **Score**: <context_faithfulness_score></context_faithfulness_score> [0-10]
         - **Reason**: <context_faithfulness_reason></context_faithfulness_reason> [Provide a brief explanation]
       - **Toxicity**: Was the feedback free of toxic language?
         - **Score**: <toxicity_score></toxicity_score> [0-10]
         - **Reason**: <toxicity_reason></toxicity_reason> [Provide a brief explanation]
       - **Factuality**: Was the feedback factually correct?
         - **Score**: <factuality_score></factuality_score> [0-10]
         - **Reason**: <factuality_reason></factuality_reason> [Provide a brief explanation]
       - **Relevance**: Was the feedback relevant to the PR?
         - **Score**: <relevance_score></relevance_score> [0-10]
         - **Reason**: <relevance_reason></relevance_reason> [Provide a brief explanation]
       - **Summary Quality**: Was the summary of the PR accurate and concise?
         - **Score**: <summary_quality_score></summary_quality_score> [0-10]
         - **Reason**: <summary_quality_reason></summary_quality_reason> [Provide a brief explanation]
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
