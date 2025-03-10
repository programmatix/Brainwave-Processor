import requests
import json
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def _get_base_url():
    base_url = os.getenv('MONGO_BASE_URL')
    if not base_url:
        raise ValueError("MONGO_BASE_URL environment variable is not set")
    return base_url

def _get_auth_credentials():
    username = os.getenv('MONGO_USERNAME')
    password = os.getenv('MONGO_PASSWORD')
    if not username or not password:
        raise ValueError("MONGO_USERNAME and MONGO_PASSWORD environment variables are not set")
    return {"username": username, "password": password}

def _fetch_with_auth(url, method='GET', headers=None, data=None):
    if headers is None:
        headers = {}
    
    # Get auth credentials from environment
    auth = _get_auth_credentials()
    
    # Create Basic auth header
    auth_string = f"{auth['username']}:{auth['password']}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    
    # Set headers
    headers['Authorization'] = f'Basic {encoded_auth}'
    headers['Content-Type'] = 'application/json'
    
    # Make the request
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        data=json.dumps(data) if data else None
    )
    
    # Check if request was successful
    response.raise_for_status()
    
    # Return JSON response
    return response.json()

def find(collection, query=None, projection=None, sort=None):
    """
    Query a MongoDB collection through the API
    
    Args:
        collection (str): The collection name
        query (dict, optional): MongoDB query. Defaults to {}.
        projection (dict, optional): Fields to include/exclude. Defaults to {}.
        sort (dict, optional): Sort specification. Defaults to None.
        
    Returns:
        list: The query results
    """
    if query is None:
        query = {}
    if projection is None:
        projection = {}
    
    data = {
        "query": query,
        "projection": projection
    }
    
    if sort is not None:
        data["sort"] = sort
    
    base_url = _get_base_url()
    return _fetch_with_auth(
        url=f"{base_url}/mongo/query/{collection}",
        method='POST',
        data=data
    )

# Example usage:
# results = find("users", {"active": True}, {"name": 1, "_id": 0}) 