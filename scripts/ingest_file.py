import json
import sys
from pathlib import Path
from cv_intelligence.ingest import ingest

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.ingest_file <resume.pdf|resume.docx>")
    print(json.dumps(ingest(Path(sys.argv[1])), indent=2))
