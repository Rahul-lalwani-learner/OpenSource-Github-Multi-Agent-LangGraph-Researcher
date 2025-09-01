from gql import gql
from core.clients import graphql_client, github_client
from pprint import pprint

def pygithub_search_repositories(query: str, top_K: int):
    """
    Search GitHub repositories using PyGithub for basic queries.
    Args:
        query (str): The search query string.
        top_K (int): Number of top repositories to return.
    Returns:
        list: List of repository objects with basic details.
    """
    
    repos = github_client.search_repositories(query=query)
    results = []
    for i, repo in enumerate(repos):
        if i >= top_K:
            break
        results.append({
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
        })
    return results

def search_github_repositories(query: str, top_K: int):
    """
    Search GitHub repositories using the GraphQL API for complex queries.
    Args:
        query (str): The search query string.
        top_K (int): Number of top repositories to return.
    Returns:
        dict: The result of the GraphQL query containing repository details.
    """
    query_str = """
    query($query: String!, $top_K: Int!) {
    search(query: $query, type: REPOSITORY, first: $top_K) {
        edges {
        node {
            ... on Repository {
            id
            name
            nameWithOwner
            url
            description
            createdAt
            updatedAt
            pushedAt
            isArchived
            isPrivate
            isFork
            forkCount
            stargazerCount
            watchers {
                totalCount
            }
            issues(states: OPEN) {
                totalCount
            }
            pullRequests(states: OPEN) {
                totalCount
            }
            licenseInfo {
                name
                nickname
            }
            primaryLanguage {
                name
            }
            languages(first: 5, orderBy: {field: SIZE, direction: DESC}) {
                edges {
                node {
                    name
                }
                size
                }
            }
            owner {
                login
                url
                __typename
            }
            }
        }
        }
    }
    }
    """

    gql_query = gql(query_str)
    variables = {"query": query, "top_K": top_K}
    result = graphql_client.execute(gql_query, variable_values=variables)
    return result

if __name__ == "__main__":
    query = "Open React Issues"
    top_K = 5
    # results = search_github_repositories(query, top_K)
    results = pygithub_search_repositories(query, top_K)
    pprint(results)
