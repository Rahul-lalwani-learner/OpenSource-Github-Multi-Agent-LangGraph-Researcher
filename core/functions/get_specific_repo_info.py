from pprint import pprint
from core.clients import github_client
import re

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

if __name__ == "__main__":
	result = get_repo_info_from_url("https://github.com/Rahul-lalwani-learner/excalidraw-draw-app")
	pprint(result)