# Error-Bot

Error-Bot is a tool for introducing errors into your codebase for testing purposes. It can add errors to single or multiple files, with or without committing the changes.

## Usage

### 1. Add error to a single file without committing

```bash
python error-bot/bot.py <file_path> <error_number>
```

### 2. Add error to multiple files without committing

```bash
python error-bot/bot.py <file_path1> <file_path2> <error_number> --multi-file
```

### 3. Add error to a single file and commit

```bash
python error-bot/bot.py <file_path> <error_number> --commit
```

### 4. Add error to multiple files and commit
(WIP) fixing a bug that crashs when a subset of files are only edited by bot
```bash
python error-bot/bot.py <file_path1> <file_path2> <error_number> --multi-file --commit
```

## Notes
Error and their details are stored in error-pattern_data-base.csv


