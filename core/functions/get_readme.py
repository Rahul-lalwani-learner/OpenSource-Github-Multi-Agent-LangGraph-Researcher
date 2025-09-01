from core.clients import github_client
import re

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
	

if __name__ == "__main__":
	result = get_repo_readme_from_url("https://github.com/Rahul-lalwani-learner/excalidraw-draw-app")
	print(result)