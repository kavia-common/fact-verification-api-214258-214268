import json
import os

from src.api.main import app

# PUBLIC_INTERFACE
def generate_openapi(output_dir: str = "interfaces", filename: str = "openapi.json") -> str:
    """
    Generate the OpenAPI schema and write it to interfaces/openapi.json by default.

    Returns:
        The full path to the written openapi.json
    """
    openapi_schema = app.openapi()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    return output_path


if __name__ == "__main__":
    path = generate_openapi()
    print(f"Wrote OpenAPI schema to {path}")
