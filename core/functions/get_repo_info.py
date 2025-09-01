import re
from core.clients import github_client
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
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}
    owner, repo_name = match.groups()
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        readme = repo.get_readme()
        content = readme.decoded_content.decode('utf-8')
        return {"readme_content": content}
    except Exception as e:
        return {"error": str(e)}

def get_repo_info_from_url(repo_url: str) -> dict:
    """
    Get detailed information about a GitHub repository from its direct URL using PyGithub.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Structured repository information, or error details if invalid.
    """
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}
    owner, repo_name = match.groups()
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        return {
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
    except Exception as e:
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
    details = {}
    details["basic_info"] = get_repo_info_from_url(repo_url)
    details["readme"] = get_repo_readme_from_url(repo_url)
    details["technologies"] = get_repo_technologies_from_url(repo_url)
    details["file_structure"] = get_repo_file_structure_from_url(repo_url, max_depth=max_depth)
    details["dependencies"] = get_repo_dependencies_from_url(repo_url)
    return details


def get_repo_technologies_from_url(repo_url: str) -> dict:
    """
    Get all technologies/languages used in a GitHub repository from its direct URL using PyGithub.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Contains 'technologies' (dict with language percentages) if available, or 'error' (str) if not found/invalid.
    """
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}
    owner, repo_name = match.groups()
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
        
        return {
            "technologies": technologies,
            "total_bytes": total_bytes
        }
    except Exception as e:
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
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}
    owner, repo_name = match.groups()
    try:
        repo = github_client.get_repo(f"{owner}/{repo_name}")
        
        def get_contents_recursive(path="", current_depth=0):
            if current_depth >= max_depth:
                return {}
            
            try:
                contents = repo.get_contents(path)
                structure = {}
                
                # Handle if contents is a single file (not a list)
                if not isinstance(contents, list):
                    contents = [contents]
                
                for content in contents:
                    if content.type == "dir":
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
            except Exception:
                return {}
        
        file_structure = get_contents_recursive()
        return {
            "file_structure": file_structure,
            "max_depth_traversed": max_depth
        }
    except Exception as e:
        return {"error": str(e)}

def get_repo_dependencies_from_url(repo_url: str) -> dict:
    """
    Get dependency files (package.json, requirements.txt, etc.) from a GitHub repository.
    Args:
        repo_url (str): The full GitHub repository URL (e.g., https://github.com/owner/repo).
    Returns:
        dict: Contains dependency information from various files, or 'error' (str) if not found/invalid.
    """
    pattern = r"https?://github.com/([^/]+)/([^/]+)(?:/)?$"
    match = re.match(pattern, repo_url)
    if not match:
        return {"error": "Invalid GitHub repository URL format."}
    owner, repo_name = match.groups()
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
            except:
                # File doesn't exist, continue to next
                continue
        
        return {
            "dependencies": dependencies,
            "found_files": list(dependencies.keys())
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    repo_url = "https://github.com/Rahul-lalwani-learner/excalidraw-draw-app"
    all_details = get_all_repo_details(repo_url, max_depth=2)
    pprint(all_details)
