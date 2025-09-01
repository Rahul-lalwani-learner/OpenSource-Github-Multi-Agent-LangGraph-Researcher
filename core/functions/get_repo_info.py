import re
from core.clients import github_client
from utils.logging import logger
import re
from pprint import pprint


def get_repo_readme_from_url(repo_url: str) -> dict:
    """
    Get the README.md content of a GitHub repository from its direct URL using PyGithub.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Contains 'readme_content' (str) if available, or 'error' (str) if not found/invalid.
    """
    logger.info(f"Fetching README for repository: {repo_url}")
    
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        logger.warning(f"Invalid GitHub repository URL format: {repo_url}")
        return {"error": "Invalid GitHub repository URL format."}
    
    owner, repo_name = match.groups()
    logger.debug(f"Extracted owner: {owner}, repo: {repo_name}")
    
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        readme = repo.get_readme()
        content = readme.decoded_content.decode('utf-8')
        
        logger.success(f"Successfully fetched README for {owner}/{repo_name}. Content length: {len(content)} characters")
        return {"readme_content": content}
    except Exception as e:
        logger.error(f"Error fetching README for {owner}/{repo_name}: {e}")
        return {"error": str(e)}

def get_repo_info_from_url(repo_url: str) -> dict:
    """
    Get detailed information about a GitHub repository from its direct URL using PyGithub.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Structured repository information, or error details if invalid.
    """
    logger.info(f"Fetching repository info for: {repo_url}")
    
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        logger.warning(f"Invalid GitHub repository URL format: {repo_url}")
        return {"error": "Invalid GitHub repository URL format."}
    
    owner, repo_name = match.groups()
    logger.debug(f"Extracted owner: {owner}, repo: {repo_name}")
    
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        
        repo_info = {
            "id": repo.id,
            "name": repo.name,
            "full_name": repo.full_name,
            "url": repo.html_url,
            "description": repo.description,
            "created_at": str(repo.created_at),
            "updated_at": str(repo.updated_at),
            "pushed_at": str(repo.pushed_at),
            "is_archived": repo.archived,
            "is_private": repo.private,
            "is_fork": repo.fork,
            "fork_count": repo.forks,
            "stargazer_count": repo.stargazers_count,
            "owner": {
                "login": repo.owner.login,
                "url": repo.owner.html_url
            }
        }
        
        logger.success(f"Successfully fetched repository info for {owner}/{repo_name}")
        return repo_info
    except Exception as e:
        logger.error(f"Error fetching repository info for {owner}/{repo_name}: {e}")
        return {"error": str(e)}
def get_all_repo_details(repo_url: str, max_depth: int = 3) -> dict:
    """
    Aggregate all available details about a GitHub repository from its URL using all repo info functions.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
        max_depth (int): Maximum depth for file structure traversal.
    Returns:
        dict: All available details about the repository.
    """
    logger.info(f"Starting comprehensive repository analysis for: {repo_url}")
    
    details = {}
    
    # Get basic repository information
    logger.debug("Fetching basic repository information...")
    details["basic_info"] = get_repo_info_from_url(repo_url)
    
    # Get README content
    logger.debug("Fetching README content...")
    details["readme"] = get_repo_readme_from_url(repo_url)
    
    # Get technologies/languages used
    logger.debug("Fetching technologies and languages...")
    details["technologies"] = get_repo_technologies_from_url(repo_url)
    
    # Get file structure
    logger.debug(f"Fetching file structure (max_depth: {max_depth})...")
    details["file_structure"] = get_repo_file_structure_from_url(repo_url, max_depth=max_depth)
    
    # Get dependency files
    logger.debug("Fetching dependency files...")
    details["dependencies"] = get_repo_dependencies_from_url(repo_url)
    
    # Count successful operations
    successful_ops = sum(1 for section in details.values() if not section.get("error"))
    total_ops = len(details)
    
    logger.success(f"Comprehensive repository analysis completed for {repo_url}. Success rate: {successful_ops}/{total_ops} operations")
    
    return details


def get_repo_technologies_from_url(repo_url: str) -> dict:
    """
    Get all technologies/languages used in a GitHub repository from its direct URL using PyGithub.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Contains 'technologies' (dict with language percentages) if available, or 'error' (str) if not found/invalid.
    """
    logger.info(f"Fetching technologies for repository: {repo_url}")
    
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        logger.warning(f"Invalid GitHub repository URL format: {repo_url}")
        return {"error": "Invalid GitHub repository URL format."}
    
    owner, repo_name = match.groups()
    logger.debug(f"Extracted owner: {owner}, repo: {repo_name}")
    
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        languages = repo.get_languages()
        total_bytes = sum(languages.values())
        
        # Calculate percentages
        technologies = {}
        if total_bytes > 0:
            for language, bytes_count in languages.items():
                percentage = round((bytes_count / total_bytes) * 100, 2)
                technologies[language] = {
                    "bytes": bytes_count,
                    "percentage": percentage
                }
        
        logger.success(f"Successfully fetched technologies for {owner}/{repo_name}. Found {len(technologies)} languages, total bytes: {total_bytes}")
        return {
            "technologies": technologies,
            "total_bytes": total_bytes
        }
    except Exception as e:
        logger.error(f"Error fetching technologies for {owner}/{repo_name}: {e}")
        return {"error": str(e)}

def get_repo_file_structure_from_url(repo_url: str, max_depth: int = 3) -> dict:
    """
    Get the file structure of a GitHub repository from its direct URL using PyGithub.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
        max_depth (int): Maximum depth to traverse the directory structure (default: 3).
    Returns:
        dict: Contains 'file_structure' (nested dict) if available, or 'error' (str) if not found/invalid.
    """
    logger.info(f"Fetching file structure for repository: {repo_url} with max_depth: {max_depth}")
    
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        logger.warning(f"Invalid GitHub repository URL format: {repo_url}")
        return {"error": "Invalid GitHub repository URL format."}
    
    owner, repo_name = match.groups()
    logger.debug(f"Extracted owner: {owner}, repo: {repo_name}")
    
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        
        def get_contents_recursive(path="", current_depth=0):
            if current_depth >= max_depth:
                logger.debug(f"Reached max depth {max_depth} at path: {path}")
                return {}
            
            try:
                contents = repo.get_contents(path)
                structure = {}
                
                # Handle if contents is a single file (not a list)
                if not isinstance(contents, list):
                    contents = [contents]
                
                for content in contents:
                    if content.type == "dir":
                        logger.debug(f"Processing directory: {content.path}")
                        structure[content.name] = {
                            "type": "directory",
                            "path": content.path,
                            "children": get_contents_recursive(content.path, current_depth + 1)
                        }
                    else:
                        structure[content.name] = {
                            "type": "file",
                            "path": content.path,
                            "size": content.size,
                            "download_url": content.download_url
                        }
                return structure
            except Exception as e:
                logger.warning(f"Error processing path '{path}': {e}")
                return {}
        
        file_structure = get_contents_recursive()
        
        # Count total files and directories
        def count_items(structure):
            files, dirs = 0, 0
            for item in structure.values():
                if item["type"] == "file":
                    files += 1
                elif item["type"] == "directory":
                    dirs += 1
                    sub_files, sub_dirs = count_items(item.get("children", {}))
                    files += sub_files
                    dirs += sub_dirs
            return files, dirs
        
        total_files, total_dirs = count_items(file_structure)
        logger.success(f"Successfully fetched file structure for {owner}/{repo_name}. Files: {total_files}, Directories: {total_dirs}")
        
        return {
            "file_structure": file_structure,
            "max_depth_traversed": max_depth,
            "total_files": total_files,
            "total_directories": total_dirs
        }
    except Exception as e:
        logger.error(f"Error fetching file structure for {owner}/{repo_name}: {e}")
        return {"error": str(e)}

def get_repo_dependencies_from_url(repo_url: str) -> dict:
    """
    Get dependency files (package.json, requirements.txt, etc.) from a GitHub repository.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Contains dependency information from various files, or 'error' (str) if not found/invalid.
    """
    logger.info(f"Fetching dependencies for repository: {repo_url}")
    
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        logger.warning(f"Invalid GitHub repository URL format: {repo_url}")
        return {"error": "Invalid GitHub repository URL format."}
    
    owner, repo_name = match.groups()
    logger.debug(f"Extracted owner: {owner}, repo: {repo_name}")
    
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        dependencies = {}
        
        # Common dependency files to check
        dependency_files = [
            "package.json",
            "requirements.txt", 
            "Pipfile",
            "pyproject.toml",
            "composer.json",
            "Gemfile",
            "pom.xml",
            "build.gradle",
            "Cargo.toml",
            "go.mod"
        ]
        
        for file_name in dependency_files:
            try:
                logger.debug(f"Checking for dependency file: {file_name}")
                file_content = repo.get_contents(file_name)
                # Handle if get_contents returns a list (shouldn't for files, but just in case)
                if isinstance(file_content, list):
                    file_content = file_content[0]
                
                content = file_content.decoded_content.decode('utf-8')
                dependencies[file_name] = {
                    "content": content,
                    "size": file_content.size,
                    "path": file_content.path
                }
                logger.debug(f"Found dependency file: {file_name} (size: {file_content.size} bytes)")
            except:
                # File doesn't exist, continue to next
                logger.debug(f"Dependency file not found: {file_name}")
                continue
        
        logger.success(f"Successfully fetched dependencies for {owner}/{repo_name}. Found {len(dependencies)} dependency files: {list(dependencies.keys())}")
        return {
            "dependencies": dependencies,
            "found_files": list(dependencies.keys())
        }
    except Exception as e:
        logger.error(f"Error fetching dependencies for {owner}/{repo_name}: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    repo_url = "https://github.com/Rahul-lalwani-learner/excalidraw-draw-app"
    all_details = get_all_repo_details(repo_url, max_depth=2)
    pprint(all_details)
