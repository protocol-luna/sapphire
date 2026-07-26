import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn
from sapphire.server import app

if __name__ == "__main__":
    port = int(os.environ.get("SAPPHIRE_PORT", "3123"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
