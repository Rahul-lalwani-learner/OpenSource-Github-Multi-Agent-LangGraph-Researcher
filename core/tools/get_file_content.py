"""
Tool to fetch the contents of a specific file from a GitHub repository given its URL and filename.
"""


from core.clients import github_client
from utils.logging import logger
import re
from langchain_core.tools import tool

@tool("get_file_content", return_direct=False)
def get_file_content_tool(repo_url: str, file_path: str) -> str:
    """
    Fetch the contents of a specific file from a GitHub repository.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
        file_path (str): The path to the file within the repository (e.g., 'README.md', 'src/main.py').
    Returns:
        str: The file content if found, or an error message if not found/invalid.
    """
    logger.info(f"Fetching file '{file_path}' from repository: {repo_url}")
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        logger.warning(f"Invalid GitHub repository URL format: {repo_url}")
        return f"Error: Invalid GitHub repository URL format."
    owner, repo_name = match.groups()
    logger.debug(f"Extracted owner: {owner}, repo: {repo_name}")
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        file_content = repo.get_contents(file_path)
        # Handle if get_contents returns a list (shouldn't for files, but just in case)
        if isinstance(file_content, list):
            file_content = file_content[0]
        content = file_content.decoded_content.decode('utf-8')
        logger.success(f"Successfully fetched file '{file_path}' from {owner}/{repo_name}")
        return content
    except Exception as e:
        logger.error(f"Error fetching file '{file_path}' from {owner}/{repo_name}: {e}")
        return f"Error: {str(e)}"
