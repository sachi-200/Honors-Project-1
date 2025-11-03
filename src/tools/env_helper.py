import os
import re
from typing import Dict

def _parse_dotenv(dotenv_path: str) -> Dict[str, str]:
    """Parse a .env file and return a dict of key -> value.

    This is a small, self-contained parser that supports:
    - lines like KEY=VALUE and export KEY=VALUE
    - quoted values with single or double quotes
    - inline comments after unquoted values
    - ignores blank lines and full-line comments starting with '#'
    """
    env: Dict[str, str] = {}
    if not os.path.exists(dotenv_path):
        return env

    line_re = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")

    with open(dotenv_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = line_re.match(line)
            if not m:
                # ignore unparseable lines
                continue
            key, val = m.group(1), m.group(2)

            # Handle quoted values
            if len(val) >= 2 and ((val[0] == val[-1]) and val[0] in ('"', "'")):
                # remove surrounding quotes and unescape common escapes
                quote = val[0]
                inner = val[1:-1]
                inner = inner.replace(f"\\{quote}", quote).replace("\\\\", "\\")
                parsed_val = inner
            else:
                # remove inline comments for unquoted values
                if "#" in val:
                    val = val.split("#", 1)[0].strip()
                parsed_val = val

            env[key] = parsed_val

    return env


def load_api_key(key_name: str = "GEMINI_API_KEY") -> str:
    """Return an API key by checking the environment then parsing a .env file.

    Behavior:
    - If the key exists in os.environ, that value is returned.
    - Otherwise, a `.env` file at the project root is parsed (but not loaded
      into the process environment) and the key is returned if present.
    - If the key is not found, a ValueError is raised with instructions.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dotenv_path = os.path.join(project_root, ".env")

    # Prefer already-exported environment variables first
    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    # Parse .env without modifying os.environ
    parsed = _parse_dotenv(dotenv_path)
    api_key = parsed.get(key_name)

    if not api_key:
        raise ValueError(
            f"'{key_name}' not found. "
            f"Please create a .env file in the project root ({project_root}) "
            f"and add the line: {key_name}='Your_API_Key_Here'"
        )

    return api_key