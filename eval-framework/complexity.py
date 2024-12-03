"""
Diff Complexity Analyzer
=======================

This tool analyzes the complexity of code changes in diff files, providing various metrics
to help assess the impact and complexity of changes in pull requests.

Features:
- Supports multiple programming languages (TypeScript, JavaScript, Python, Java, Go, Rust, C/C++, PHP)
- Analyzes complexity metrics including:
  * Lines of code (LOC)
  * Additions and deletions
  * Cyclomatic complexity
  * Cognitive complexity
  * Halstead metrics
  * Maximum nesting depth
  * Token count

Usage:
1. Place your diff content in a file (e.g., 'diff/diff4.txt') between <diff_to_summarize> tags:
   <diff_to_summarize>
   # Diff to summarize for `path/to/file.ext`
   ```diff
   your diff content here
   ```
   </diff_to_summarize>

2. Run the script:
   python smith/complexity.py

The script will output detailed complexity metrics for the provided diff.

Example output:
    Analysis Results for path/to/file.ext
    Language detected: python
    
    Complexity Metrics:
    - lines_of_code: 42
    - additions: 30
    - deletions: 12
    - cyclomatic_complexity: 5
    ...
"""

import re
import os
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum
from pygments.lexers import guess_lexer, get_lexer_for_filename
from pygments.util import ClassNotFound

class ChangeType(Enum):
    ADDITION = 'addition'
    DELETION = 'deletion'
    CONTEXT = 'context'

@dataclass
class DiffLine:
    content: str
    type: ChangeType
    line_number: int

@dataclass
class DiffHunk:
    file_path: str
    language: str
    lines: List[DiffLine]

class ComplexityMetrics:
    def __init__(self):
        self.loc: int = 0  # Lines of code
        self.additions: int = 0  # Number of added lines
        self.deletions: int = 0  # Number of deleted lines
        self.cyclomatic: int = 0  # Cyclomatic complexity
        self.cognitive: int = 0  # Cognitive complexity (future improvement)
        self.halstead: Dict = {}  # Halstead metrics
        self.nesting_depth: int = 0  # Maximum nesting depth
        self.token_count: int = 0  # Token count for complexity analysis

    def to_dict(self) -> Dict:
        return {
            'lines_of_code': self.loc,
            'additions': self.additions,
            'deletions': self.deletions,
            'cyclomatic_complexity': self.cyclomatic,
            'cognitive_complexity': self.cognitive,
            'max_nesting_depth': self.nesting_depth,
            'token_count': self.token_count,
            'halstead_metrics': self.halstead
        }

class DiffAnalyzer:
    def __init__(self):
        self.supported_languages = {
            'typescript': ['ts', 'tsx'],
            'javascript': ['js', 'jsx'],
            'python': ['py'],
            'java': ['java'],
            'go': ['go'],
            'rust': ['rs'],
            'c': ['c'],
            'cpp': ['cpp', 'cxx', 'cc'],
            'php': ['php']
        }

    def read_diff_file(self, file_path: str) -> Optional[str]:
        """Read diff content from file with tags."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                # Extract content between <diff_to_summarize> tags
                match = re.search(r'<diff_to_summarize>(.*?)</diff_to_summarize>', 
                                content, re.DOTALL)
                if match:
                    # Convert literal \n to actual newlines
                    return match.group(1).replace('\\n', '\n')
                return None
        except FileNotFoundError:
            print(f"Error: {file_path} not found")
            return None

    def parse_diff_text(self, diff_text: str) -> Optional[DiffHunk]:
        """Parse the diff text and extract relevant information."""
        if not diff_text:
            return None

        # Extract file path
        file_match = re.search(r'#\s*Diff to summarize for `([^`]+)`', diff_text)
        if not file_match:
            return None

        file_path = file_match.group(1)

        # Extract the diff content
        diff_content_match = re.search(r'```diff\n(.*?)\n```', diff_text, re.DOTALL)
        if not diff_content_match:
            return None

        diff_content = diff_content_match.group(1)

        # Detect language
        extension = file_path.split('.')[-1]
        language = self._detect_language(extension, diff_content)

        # Parse lines
        lines = self._parse_lines(diff_content)

        return DiffHunk(file_path, language, lines)

    def _detect_language(self, extension: str, content: str) -> str:
        """Detect the programming language based on extension and content."""
        for lang, exts in self.supported_languages.items():
            if extension in exts:
                return lang

        try:
            lexer = guess_lexer(content)
            return lexer.name.lower()
        except ClassNotFound:
            return 'unknown'

    def _parse_lines(self, content: str) -> List[DiffLine]:
        """Parse the diff content into structured line objects."""
        lines = []
        line_number = 1

        for line in content.splitlines():
            if not line.strip():
                continue

            if line.startswith('+'):
                change_type = ChangeType.ADDITION
                line_content = line[1:]
            elif line.startswith('-'):
                change_type = ChangeType.DELETION
                line_content = line[1:]
            else:
                change_type = ChangeType.CONTEXT
                line_content = line

            lines.append(DiffLine(line_content.strip(), change_type, line_number))
            line_number += 1

        return lines

    def analyze_complexity(self, hunk: DiffHunk) -> ComplexityMetrics:
        """Analyze the complexity of the diff hunk."""
        metrics = ComplexityMetrics()

        # Basic metrics
        metrics.loc = len([l for l in hunk.lines if l.content.strip()])
        metrics.additions = len([l for l in hunk.lines if l.type == ChangeType.ADDITION])
        metrics.deletions = len([l for l in hunk.lines if l.type == ChangeType.DELETION])

        # Language-specific analysis
        if hunk.language in ['typescript', 'javascript']:
            self._analyze_ts_js(hunk, metrics)
        else:
            self._analyze_generic(hunk, metrics)

        return metrics

    def _analyze_ts_js(self, hunk: DiffHunk, metrics: ComplexityMetrics):
        """Specific analysis for TypeScript/JavaScript."""
        self._analyze_control_flow(hunk, metrics)

    def _analyze_generic(self, hunk: DiffHunk, metrics: ComplexityMetrics):
        """Generic analysis for all languages including unsupported ones."""
        self._analyze_control_flow(hunk, metrics)

    def _analyze_control_flow(self, hunk: DiffHunk, metrics: ComplexityMetrics):
        """Analyze control flow to determine cyclomatic complexity for all languages."""
        operators = set()
        operands = set()
        nesting_level = 0
        max_nesting = 0

        # Common control flow patterns across multiple languages
        control_flow_patterns = [
            r'\bif\b', r'\belse\b', r'\bswitch\b', r'\bcase\b',
            r'\bfor\b', r'\bwhile\b', r'\bdo\b', r'\bcatch\b',
            r'\btry\b', r'\breturn\b', r'\bthrow\b'
        ]

        for line in hunk.lines:
            content = line.content.strip()

            if not content:
                continue

            if line.type != ChangeType.DELETION:
                # Nesting analysis
                nesting_delta = content.count('{') - content.count('}')
                nesting_level += nesting_delta
                max_nesting = max(max_nesting, nesting_level)

                # Complexity metrics
                for pattern in control_flow_patterns:
                    if re.search(pattern, content):
                        metrics.cyclomatic += 1

                # Collect Halstead metrics
                operators.update(re.findall(r'[+\-*/%=&|<>!]+', content))
                operands.update(re.findall(r'\b\w+\b', content))

                # Update token count (new way of measuring complexity)
                metrics.token_count += len(re.findall(r'\S+', content))

        # Update metrics
        metrics.nesting_depth = max_nesting
        metrics.halstead = self._compute_halstead(operators, operands)

    def _compute_halstead(self, operators: Set[str], operands: Set[str]) -> Dict:
        """Compute Halstead metrics."""
        n1 = len(operators)
        n2 = len(operands)
        N1 = sum(1 for _ in operators)
        N2 = sum(1 for _ in operands)

        try:
            vocabulary = n1 + n2
            length = N1 + N2
            volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
            difficulty = (n1 * N2) / (2 * n2) if n2 > 0 else 0
            effort = difficulty * volume
        except ZeroDivisionError:
            return {}

        return {
            'vocabulary': vocabulary,
            'length': length,
            'volume': volume,
            'difficulty': difficulty,
            'effort': effort
        }

def compare_pr_metrics(metrics1: ComplexityMetrics, metrics2: ComplexityMetrics) -> Dict[str, str]:
    """Compare complexity metrics between two PRs and return a comparison summary."""
    comparison = {}
    metrics1_dict = metrics1.to_dict()
    metrics2_dict = metrics2.to_dict()

    for metric in metrics1_dict:
        value1 = metrics1_dict[metric]
        value2 = metrics2_dict[metric]
        if isinstance(value1, dict):
            comparison[metric] = ""
            for sub_metric in value1:
                comparison[metric] += f"{sub_metric}: PR1={value1[sub_metric]:.2f}, PR2={value2[sub_metric]:.2f}\n"
        else:
            comparison[metric] = f"PR1={value1}, PR2={value2}"

    return comparison

def main():
    analyzer = DiffAnalyzer()

    # Read diff from file
    diff_text = analyzer.read_diff_file('diff/diff4.txt')
    if not diff_text:
        print("Error: Could not read diff content")
        print("Make sure diff.txt exists and contains the diff between <diff_to_summarize> tags")
        return

    # Parse and analyze
    hunk = analyzer.parse_diff_text(diff_text)
    if not hunk:
        print("Error: Could not parse diff")
        print("Diff text first 100 chars:", diff_text[:100])
        return

    metrics = analyzer.analyze_complexity(hunk)

    # Print results
    print(f"\nAnalysis Results for {hunk.file_path}")
    print(f"Language detected: {hunk.language}")
    print("\nComplexity Metrics:")
    for metric, value in metrics.to_dict().items():
        if isinstance(value, dict):
            print(f"\n{metric}:")
            for k, v in value.items():
                print(f"  - {k}: {v:.2f}")
        else:
            print(f"- {metric}: {value}")

if __name__ == "__main__":
    main()