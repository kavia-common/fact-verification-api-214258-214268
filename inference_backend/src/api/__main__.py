"""
Entrypoint helper to generate OpenAPI schema via: python -m src.api
This module imports the FastAPI app and writes the schema to interfaces/openapi.json.
"""

import json
import os

from src.api.main import app  # ensures routes and models are registered

def main() -> None:
    # PUBLIC_INTERFACE
    """
    Generate the OpenAPI schema to interfaces/openapi.json.

    This reads the FastAPI app's openapi() output to ensure all currently
    registered routes and models (/health, /inference/run and /inference/stream, etc.)
    are reflected in the saved spec.

    Note:
    - Do not hardcode config; this purely dumps schema.
    - Caller must ensure proper working directory (project root of container).
    """
    schema = app.openapi()
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "interfaces")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "openapi.json")
    with open(output_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Wrote OpenAPI schema to {output_path}")

if __name__ == "__main__":
    main()
