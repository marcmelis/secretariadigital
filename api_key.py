import os
import sys

try:
    API_KEY = os.environ['API_KEY']
except KeyError:
    try:
        import config
        API_KEY = config.API_KEY
    except (ImportError, AttributeError):
        print("API_KEY not found in environment variables or config.py. Please set the API key.")
        sys.exit(1)
