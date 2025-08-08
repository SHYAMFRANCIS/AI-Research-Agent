from langchain_community.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime   

def save_to_txt(data:str, filename:str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_data = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_data)

# Search tool
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="web_search",
    func=search.run,
    description="Search the web for current information on any topic. Use this when you need up-to-date information.",
)

# Save tool
save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured response data to a txt file.",
)

# Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=5,doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
