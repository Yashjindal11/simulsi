"""Print the CHANGELOG section for a version (used by the release workflow)."""

import re
import sys
from pathlib import Path

version = sys.argv[1]
text = Path(__file__).resolve().parent.parent.joinpath("CHANGELOG.md").read_text("utf-8")
match = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.M | re.S)
if not match:
    sys.exit(f"no CHANGELOG entry for {version}")
print(match.group(1).strip())
