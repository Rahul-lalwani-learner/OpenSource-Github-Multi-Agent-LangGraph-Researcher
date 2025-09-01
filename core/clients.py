"""
GitHub and GraphQL clients initialization module.
This module provides centralized client instances to avoid recreating them in every function.
"""

from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport
from github import Github
from config import settings

# Initialize GraphQL client
graphql_transport = RequestsHTTPTransport(
    url=settings.GRAPHQL_API_ENDPOINT,
    headers={"Authorization": f"Bearer {settings.GITHUB_API_KEY}"},
    use_json=True,
)
graphql_client = Client(transport=graphql_transport, fetch_schema_from_transport=False)

# Initialize PyGithub client
github_client = Github(settings.GITHUB_API_KEY)
