"""Prevent accidental addition of generated datasets to new git commits."""
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ignore_file = root / '.gitignore'
    existing = ignore_file.read_text(encoding='utf-8') if ignore_file.exists() else ''
    lines = [
        '.env',
        'data/*.csv',
        'data/download_cache/',
        'data/individual/',
        'data/ai_prompt_plan_1000.csv',
        'model/cache.sqlite3*',
    ]
    missing = [line for line in lines if line not in existing.splitlines()]
    if missing:
        with ignore_file.open('a', encoding='utf-8') as file:
            file.write(('\n' if existing and not existing.endswith('\n') else '') +
                       '\n# Kirusha private training data\n' + '\n'.join(missing) + '\n')
    print('Updated .gitignore (note: previously committed secrets remain in Git history).')


if __name__ == '__main__':
    main()
