from pathlib import Path
from dotenv import load_dotenv
import os


def load_environment_variables():
    """
    Load environment variables from a .env file located in the parent directory.
    Raises an error if the file does not exist or if required variables are missing.
    """
    env_path = Path('.').parent / '.env'
    print(f"Loading environment variables from {env_path}")
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file {env_path} not found. Please create a .env file with the required variables.")
    load_dotenv(dotenv_path=env_path)


def fetch_azure_openai_credentials():
    load_environment_variables()
    
    AZURE_OPEN_API_KEY = os.getenv("AZURE_OPEN_API_KEY")
    AZURE_OPEN_API_ENDPOINT = os.getenv("AZURE_OPEN_API_ENDPOINT")
    AZURE_OPEN_API_VERSION = os.getenv("AZURE_OPEN_API_VERSION")
    AZURE_EMBEDDING_API_ENDPOINT = os.getenv("AZURE_EMBEDDING_API_ENDPOINT")
    AZURE_EMBEDDING_VERSION = os.getenv("AZURE_EMBEDDING_VERSION")

    if not AZURE_OPEN_API_KEY or not AZURE_OPEN_API_ENDPOINT or not AZURE_OPEN_API_VERSION:
        raise ValueError("Please set the AZURE_OPEN_API_KEY, AZURE_OPEN_API_ENDPOINT, and AZURE_OPEN_API_VERSION environment variables.")
    else:
        print("Environment variables loaded successfully.")
        return AZURE_OPEN_API_KEY, AZURE_OPEN_API_ENDPOINT, AZURE_OPEN_API_VERSION, AZURE_EMBEDDING_API_ENDPOINT, AZURE_EMBEDDING_VERSION
# env_path = Path('.').parent / '.env'
# print(f"Loading environment variables from {env_path}")
# if not env_path.exists():
#     raise FileNotFoundError(f"Environment file {env_path} not found. Please create a .env file with the required variables.")
# load_dotenv(dotenv_path=env_path)

# AZURE_OPEN_API_KEY = os.getenv("AZURE_OPEN_API_KEY")
# AZURE_OPEN_API_ENDPOINT = os.getenv("AZURE_OPEN_API_ENDPOINT")
# AZURE_OPEN_API_VERSION = os.getenv("AZURE_OPEN_API_VERSION")

# if not AZURE_OPEN_API_KEY or not AZURE_OPEN_API_ENDPOINT or not AZURE_OPEN_API_VERSION:
#     raise ValueError("Please set the AZURE_OPEN_API_KEY, AZURE_OPEN_API_ENDPOINT, and AZURE_OPEN_API_VERSION environment variables.")
# else:
#     print("Environment variables loaded successfully.")


