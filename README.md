# TaxRAG

TaxRAG is a Python project designed to assist with tax-related tasks by leveraging the power of RAG (Retrieval-Augmented Generation) techniques. It provides tools for extracting, processing, and generating tax-related information from AEAT.

## Features

- **Data Extraction**: Extracts tax-related data from AEAT.
- **Data Processing**: Processes and cleans tax data for analysis.
- **Data Generation**: Generates tax-related documents and reports.

## Installation

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Scrapper

In order to run the scrapper to get the RAG data from AEAT, you have to run the `scrap.py` script. This script will scrape the necessary data from AEAT and save it in the specified directory (`renta2024` by default) for further processing.

```bash
python scrap.py <directory>
```

This script can take a while to run, depending on the amount of data to be scraped. It will create a directory named `renta2024` (or the specified directory) containing the scraped data.

### Agent

You have to provide your Hugging Face API and OpenAI API keys by setting the environment variables `HUGGINGFACE_API_KEY` and `OPENAI_API_KEY` respectively. You can do this in your terminal or command prompt:

```bash
export HUGGINGFACE_API_KEY='your_huggingface_api_key'
export OPENAI_API_KEY='your_openai_api_key'
```

To use the TaxRAG agent, you can run the main script with streamlit:

```bash
streamlit run rag.py
```

## Offline Agents

In order to run offline agents, you need to set the `USE_ONLINE_AGENTS` environment variable to `False`. This will ensure that the agents do not attempt to access online resources.

```bash
export USE_ONLINE_AGENTS=False
```

Then, you can run the offline agents running the main script with streamlit:

```bash
streamlit run rag.py
```
