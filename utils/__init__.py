import requests

from bs4 import BeautifulSoup
from urllib.parse import unquote
from serpapi import GoogleSearch

# Make Kaggle import optional - it requires authentication config
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except (ImportError, OSError):
    KaggleApi = None
    KAGGLE_AVAILABLE = False

from openai import OpenAI
from configs import AVAILABLE_LLMs, Configs


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_kaggle():
    if not KAGGLE_AVAILABLE:
        raise ImportError(
            "Kaggle API is not available. Please install kaggle and configure "
            "~/.kaggle/kaggle.json, or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
    api = KaggleApi()
    api.authenticate()
    return api


# def search_web(query):
#     try:
#         # Abort the request after 10 seconds
#         response = requests.get(f"https://www.google.com/search?hl=en&q={query}")
#         response.raise_for_status()  # Raises an HTTPError for bad responses
#         html_string = response.text
#     except requests.exceptions.RequestException as e:
#         print_message(
#             "system",
#             "Request Google Search Failed with " + str(e) + "\n Using SerpAPI.",
#         )
#         params = {
#             "engine": "google",
#             "q": query,
#             "api_key": "",
#         }

#         search = GoogleSearch(params)
#         results = search.get_dict()
#         return results["organic_results"]

#     # Parse the HTML content
#     soup = BeautifulSoup(html_string, "html.parser")

#     # Find all <a> tags
#     links = soup.find_all("a")

#     if not links:
#         raise Exception('Webpage does not have any "a" element')

#     # Filter and process the links
#     filtered_links = []
#     for link in links:
#         href = link.get("href")
#         if href and href.startswith("/url?q=") and "google.com" not in href:
#             cleaned_link = unquote(
#                 href.split("&sa=")[0][7:]
#             )  # Remove "/url?q=" and split at "&sa="
#             filtered_links.append(cleaned_link)

#     # Remove duplicates and prepare the output
#     unique_links = list(set(filtered_links))
#     return {"organic_results": [{"link": link} for link in unique_links]}[
#         "organic_results"
#     ]

def search_web(query):
    """
    Search the web using SerpAPI. Returns empty list if API key is missing or request fails.
    """
    api_key = Configs.SEARCHAPI_API_KEY
    if not api_key or api_key == "":
        print_message(
            "system",
            "SerpAPI key not configured. Web search will return empty results. "
            "Set SEARCHAPI_API_KEY environment variable to enable web search."
        )
        return []
    
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Handle API errors
        if "error" in results:
            print_message(
                "system",
                f"SerpAPI error: {results.get('error', 'Unknown error')}. "
                "Web search will return empty results."
            )
            return []
        
        # Return organic results if available, otherwise empty list
        return results.get("organic_results", [])
    except Exception as e:
        print_message(
            "system",
            f"Web search failed with error: {e}. Returning empty results."
        )
        return []


def print_message(sender, msg, pid=None):
    pid = f"-{pid}" if pid else ""
    sender_color = {
        "user": color.PURPLE,
        "system": color.RED,
        "manager": color.GREEN,
        "model": color.BLUE,
        "data": color.DARKCYAN,
        "prompt": color.CYAN,
        "operation": color.YELLOW,
    }
    sender_label = {
        "user": "💬 You:",
        "system": "⚠️ SYSTEM NOTICE ⚠️\n",
        "manager": "🕴🏻 Agent Manager:",
        "model": f"🦙 Model Agent{pid}:",
        "data": f"🦙 Data Agent{pid}:",
        "prompt": "🦙 Prompt Agent:",
        "operation": f"🦙 Operation Agent{pid}:",
    }

    msg = f"{color.BOLD}{sender_color[sender]}{sender_label[sender]}{color.END}{color.END} {msg}"
    print(msg)
    print()


def get_client(llm: str = "qwen"):
    if llm.startswith("gpt"):
        return OpenAI(api_key=AVAILABLE_LLMs[llm]["api_key"])
    else:
        return OpenAI(
            base_url=AVAILABLE_LLMs[llm]["base_url"],
            api_key=AVAILABLE_LLMs[llm]["api_key"],
        )
