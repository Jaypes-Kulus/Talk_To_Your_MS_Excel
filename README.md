# Excel SQL Chatbot

This project provides a Gradio-powered chatbot interface to interact with your Excel data using natural language. The chatbot converts your questions into SQL queries and returns results and insights from your uploaded Excel file.

## Features

- Upload an Excel file (`.xlsx` or `.xls`)
- Select a sheet from your file
- Optionally provide a custom table description (recommended for best results)
- Ask questions in natural language about your data
- Get SQL queries, results, and concise analysis

## Setup

1. **Clone this repository** and navigate to the project folder.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key**:
   - Create a `.env` file in the project directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

Run the Gradio app with:
```bash
gradio gradio_interface.py
```
Or:
```bash
python gradio_interface.py
```

## Instructions

- Ensure that column names are present in the first row of your sheet. This is required for proper SQL conversion and accurate analysis.
- Upload your Excel file (`.xlsx` or `.xls`).
- Select the sheet you want to analyze.
- Provide a table description similar to:
  ```
  The table named data contains student marks with columns: Name, Math, Science, English, Location
  ```
  *(You can leave this blank to use the auto-generated description.)*
- Ask questions about your data in natural language, such as:
  - "Show me the first 10 rows"
  - "What's the average score in Math?"
  - "List all unique locations"

## Notes

- The table name in SQL queries is always `data`.
- Only Excel files are supported.
- Make sure your `.env` file contains a valid OPENAI_API_ key.