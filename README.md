OpenSource-Github-Multi-Agent-LangGraph-Researcher
==================================================

## Overview

This project is a multi-agent research and automation framework for GitHub repositories, built using LangGraph, LangChain, and advanced search tools. It enables agents to search, analyze, and extract structured information from any public GitHub repository, leveraging both web search APIs and direct GitHub API access.

## Features

- **Modular Python Package Structure**: Organized into `core`, `config`, `utils`, and `tools` for maintainability and scalability.
- **Centralized Client Management**: All API clients (GitHub, GraphQL) are initialized in `core/clients.py` for efficient reuse.
- **Comprehensive Repo Analysis**: Functions to fetch repo info, README, technologies, file structure, and dependencies from a single URL.
- **LangGraph Tool Integration**: Plug-and-play search tools for Serper (Google-like) and Tavily (research-oriented) queries.
- **Advanced Logging**: All modules use Loguru-based logging for debugging, monitoring, and error tracking.
- **Environment-Based Configuration**: API keys and settings managed via `.env` and `config/settings.py`.

## Main Components

- `core/clients.py`: Initializes and shares GitHub and GraphQL clients.
- `core/functions/get_repo_info.py`: Unified functions to extract all details about a GitHub repo (info, README, tech, structure, dependencies).
- `core/functions/graphql_search_repo.py`: Search GitHub repos using GraphQL or PyGithub, with docstrings and logging.
- `core/tools/search_tools.py`: LangGraph-compatible tools for Serper and Tavily search APIs.
- `utils/logging.py`: Loguru-based logging setup.
- `config/settings.py`: Pydantic-based settings loader for API keys and endpoints.

## Search Tools

- **Serper**: Best for broad, real-time web search (trends, news, general queries).
- **Tavily**: Best for deep, factual, research-oriented queries (technical docs, academic info).

## Example Usage

```python
from core.functions.get_repo_info import get_all_repo_details
repo_url = "https://github.com/Rahul-lalwani-learner/excalidraw-draw-app"
details = get_all_repo_details(repo_url)
print(details)

from core.tools import serper_search_tool, tavily_search_tool
print(serper_search_tool("latest AI frameworks 2025"))
print(tavily_search_tool("explain quantum AI in simple terms"))
```

## Logging

All logs are saved to `logs/app_{time}.log` with rotation and retention. You can monitor agent actions, errors, and API usage.

## Configuration

Set your API keys in a `.env` file:

```
GITHUB_API_KEY=your_github_token
SERPER_API_SECRET=your_serper_key
TAVILY_API_SECRET=your_tavily_key
GEMINI_API_KEY=your_gemini_key
```

## Requirements

All dependencies are managed in `requirements.in` and compiled with `pip-compile`.

## Extending

- Add new tools in `core/tools/` and register them in `__init__.py`.
- Add new repo analysis functions in `core/functions/get_repo_info.py`.

## License

MIT

