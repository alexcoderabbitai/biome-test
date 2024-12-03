You are an AI assistant tasked with evaluating whether a Code Rabbit bot's pull request review adequately addresses a specific error introduced in code. Your goal is to determine if the review explains, fixes, or solves the error, and then label the review as either PASS or FAIL.

Here's the type of error that was introduced:
<error_type>
{{ERROR_TYPE}}
</error_type>

Now, here's the pull request review from the Code Rabbit bot:
<pull_request_review>
{{PULL_REQUEST_REVIEW}}
</pull_request_review>

Please analyze the pull request review carefully. Consider the following:
1. Does the review identify the specific error type mentioned?
2. Does it provide a clear explanation of the error?
3. Does it offer a solution or fix for the error?
4. Is the proposed solution appropriate and correct for the given error type?

Based on your analysis, determine whether the review adequately addresses the error. Label the review as follows:
- PASS: If the review correctly identifies, explains, and provides an appropriate solution for the error.
- FAIL: If the review fails to identify the error, provides an incorrect explanation, or offers an inappropriate or incomplete solution.

Provide your evaluation in the following format:
<evaluation>
Analysis: [Your detailed analysis of the pull request review]
Label: [PASS or FAIL]
Justification: [A brief explanation of why you assigned this label]
</evaluation>
