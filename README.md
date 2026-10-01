# AI Research Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Tool--Calling%20Agent-green)](https://www.langchain.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-orange?logo=google)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An AI-powered research assistant that answers research questions with structured topic summaries and cited sources. A Python backend built on **LangChain** and **Google Gemini 2.0 Flash** orchestrates web-search and Wikipedia tools through a tool-calling agent, with production-grade retry logic, circuit breakers, and structured error handling. A lightweight **HTML/CSS/JS frontend** lets you submit queries and view results.

## Features

- **Agentic research** — LangChain tool-calling agent (`create_tool_calling_agent` + `AgentExecutor`) plans which tools to use per query (`main.py`).
- **Structured output** — Responses parsed via `PydanticOutputParser` into `topic`, `summary`, `sources`, `tools_used` (`ResponseModel`).
- **Web search tool** — `DuckDuckGoSearchRun` for current, up-to-date information (`tools.py`).
- **Wikipedia tool** — `WikipediaQueryRun` (top 5 results) for background knowledge (`tools.py`).
- **Save-to-file tool** — Appends timestamped research output to `research_output.txt` (`tools.py`).
- **Resilient execution** — Exponential-backoff retries with jitter, per-service retry configs, and circuit breakers for the Gemini API and LangChain agent (`retry_handler.py`).
- **Structured errors** — Custom exception hierarchy (`ApiQuotaError`, `NetworkError`, `ParsingError`, …), user-friendly messages with recovery hints, and session error summaries (`exceptions.py`, `error_handler.py`, `error_logger.py`).
- **Simple web UI** — Query form that renders the topic summary and source list (`index.html`, `app.js`, `styles.css`).
- **Tested** — Unit, comprehensive, and integration-scenario suites for the error-handling stack (`test_error_handling.py`, `test_comprehensive_error_handling.py`, `test_integration_scenarios.py`).

## Tech Stack

| Layer    | Technology |
|----------|------------|
| LLM      | Google Gemini 2.0 Flash (`langchain-google-genai`) |
| Agent    | LangChain tool-calling agent, Pydantic output parsing |
| Tools    | DuckDuckGo Search, Wikipedia API |
| Backend  | Python, Flask + Flask-CORS |
| Frontend | HTML, CSS, vanilla JavaScript |
| Config   | `python-dotenv` (`.env`) |

## Project Structure

```text
AI-Research-Agent/
├── main.py                            # Agent setup, research workflow, Flask app
├── tools.py                           # web_search, wiki, save_text_to_file tools
├── error_handler.py                   # Global handler, error contexts, user messages
├── exceptions.py                      # Custom exception hierarchy
├── error_logger.py                    # Structured logging
├── retry_handler.py                   # Retry configs, backoff, circuit breakers
├── app.js                             # Frontend: submits query, renders summary+sources
├── index.html                         # Frontend: query form + response area
├── styles.css                         # Frontend styles
├── requierments.txt                   # Python dependencies (note filename spelling)
├── template/                          # Templates
├── test_error_handling.py             # Error-handling unit tests
├── test_comprehensive_error_handling.py
├── test_integration_scenarios.py
├── ERROR_HANDLING_STRATEGY.md         # Error-handling design notes
└── error_handling_matrix.md
```

## Installation

**Prerequisites:** Python 3.8+ and a [Google AI Studio](https://ai.google.dev/) API key.

```bash
git clone https://github.com/SHYAMFRANCIS/AI-Research-Agent.git
cd AI-Research-Agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requierments.txt
pip install flask flask-cors    # required by main.py (not in requirements file)
```

Configure your API key in a `.env` file (a `.env` containing `GOOGLE_API_KEY` is expected):

```env
GOOGLE_API_KEY=your_api_key_here
```

## Usage

Run the research assistant from the command line:

```bash
python main.py
```

You will be prompted for a research topic:

```text
What can i help you research? <your question>
```

The agent searches the web and Wikipedia, then prints a structured summary with sources. Results are also appended to `research_output.txt`.

### Frontend

`app.js` submits queries via `POST http://localhost:5000/api/research` with body `{"query": "..."}` and renders `topic`, `summary`, and `sources`. Serve `index.html` (e.g. `python -m http.server`) once the backend exposes that endpoint.

### Examples

```text
What can i help you research? What are the latest advances in quantum error correction?
```

```json
{
  "topic": "Quantum Error Correction",
  "summary": "...",
  "sources": ["https://..."],
  "tools_used": ["web_search", "wikipedia"]
}
```

### Running the tests

```bash
python test_error_handling.py
python test_comprehensive_error_handling.py
python test_integration_scenarios.py
```

## Configuration / Environment

| Variable         | Description                              |
|------------------|------------------------------------------|
| `GOOGLE_API_KEY` | Google AI Studio key for Gemini 2.0 Flash |

Retry behaviour (attempts, delays, circuit-breaker thresholds) is tuned in `main.py` via `RetryConfig` and `create_circuit_breaker`.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Add or update tests for error paths where relevant.
4. Open a pull request with a clear description.

## License

No `LICENSE` file is present in this repository. The code is shared publicly by the author; if you intend to reuse it, please confirm licensing with the repository owner. (This README defaults to referencing MIT — a `LICENSE` file should be added to make that explicit.)
