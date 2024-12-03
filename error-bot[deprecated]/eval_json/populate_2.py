import json

def process_pr_data(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create new list to store processed PRs
    processed_prs = []
    
    # Process each PR
    for pr in data:
        processed_pr = {
            # Basic PR Information
            'id': pr.get('id'),                      # String: "3"
            'type': pr.get('type'),                  # String: "1"
            'number': pr.get('number'),              # Integer: 52
         #   'base_branch': pr.get('base_branch'),    # String: "main-copy"
            'head_branch': pr.get('head_branch'),    # String: "error-010-api"
         #   'title': pr.get('title'),                # String: "Automated Test: @eval-junior-reviewer-1"
         #   'user': pr.get('user'),                  # String: "nehal-a2z"
         #   'state': pr.get('state'),                # String: "closed"
            
            # Timestamps
         #   'created_at': pr.get('created_at'),      # String: "2024-10-24T06:13:41+00:00"
         #   'closed_at': pr.get('closed_at'),        # String: "2024-10-24T06:15:02+00:00"
         #   'merged_at': pr.get('merged_at'),        # Null or timestamp
            
            # Git Information
         #   'merge_commit_sha': pr.get('merge_commit_sha'),  # String: "38bb3d43fea6da..."
            
            # PR Content
         #   'body': pr.get('body'),                  # String: "This pull request..."
         #   'labels': pr.get('labels'),              # Array: []
            
            # PR Statistics
         #   'commits': pr.get('commits'),            # Integer: 1
        #    'additions': pr.get('additions'),        # Integer: 3
        #    'deletions': pr.get('deletions'),        # Integer: 0
         #   'changed_files': pr.get('changed_files'), # Integer: 1
            
            # Comments and Reviews
            'comments': [                            # Array of comment objects
                {
          #          'user': comment.get('user'),           # String: "coderabbitai[bot]"
           #         'created_at': comment.get('created_at'), # String: timestamp
                    'body': comment.get('body')            # String: comment content
                }
                for comment in pr.get('comments', [])
            ],
            
            'review_comments': [                     # Array of review comment objects
                {
            #        'user': comment.get('user'),
            #        'created_at': comment.get('created_at'),
                    'body': comment.get('body'),
                    'path': comment.get('path'),           # String: file path
                    'position': comment.get('position'),   # Integer: line position
            #        'line_range': comment.get('line_range') # String: line range
                }
                for comment in pr.get('review_comments', [])
            ],
            
            # 'reviews': [                            # Array of review objects
            #     {
            #         'user': review.get('user'),
            #         'state': review.get('state'),         # String: "COMMENTED"
            #         'submitted_at': review.get('submitted_at'),
            #         'body': review.get('body')
            #     }
            #     for review in pr.get('reviews', [])
            # ],
            
          #  'requested_reviewers': pr.get('requested_reviewers'),  # Array
            
            # Code Changes
          #  'hunks': pr.get('hunks'),                # Array of code hunks
            
            # Commit Information
            # 'commits_data': [                        # Array of commit objects
            #     {
            #         'sha': commit.get('sha'),             # String: commit hash
            #         'author': commit.get('author'),       # String: author name
            #         'message': commit.get('message'),     # String: commit message
            #         'date': commit.get('date'),          # String: timestamp
            #         'files_changed': [                    # Array of changed files
            #             {
            #                 'filename': file.get('filename'),
            #                 'additions': file.get('additions'),
            #                 'deletions': file.get('deletions'),
            #                 'changes': file.get('changes'),
            #                 'status': file.get('status')
            #             }
            #             for file in commit.get('files_changed', [])
            #         ]
            #     }
            #     for commit in pr.get('commits_data', [])
            # ],
            
            # File Changes
            'file_changes': [                        # Array of file change objects
                {
                    'file': change.get('file'),          # String: file path
                    'hunks': [                           # Array of code hunks
                        {
                            'id': hunk.get('id'),
             #              'old_count': hunk.get('old_count'),
              #              'new_start': hunk.get('new_start'),
               #             'new_count': hunk.get('new_count'),
                            'content': hunk.get('content')
                        }
                        for hunk in change.get('hunks', [])
                    ]
                }
                for change in pr.get('file_changes', [])
            ]
        }
        processed_prs.append(processed_pr)
    
    # Write to output file
    with open(output_file, 'w') as f:
        json.dump(processed_prs, f, indent=4)

# Run the processor
process_pr_data('pr_tpye_8.json', 'processed_prs_8.json')
