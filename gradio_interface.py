import gradio as gr
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import pandas as pd
from sqlalchemy import create_engine
from utils import read_excel_file, execute_query
from tabulate import tabulate
import os
import tempfile

load_dotenv()  # Load environment variables from .env file

# Global variables to store current data
current_df = None
current_engine = None
current_table_description = ""

def load_excel_data(file_path, sheet_name=None):
    """Load Excel file and return DataFrame with column information"""
    try:
        if sheet_name and sheet_name != "Select Sheet":
            df = read_excel_file(file_path, sheet_name=sheet_name)
        else:
            df = read_excel_file(file_path)
        
        # Generate automatic table description
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).tolist()
            columns_info.append(f"{col} ({dtype}): {sample_values}")
        
        auto_description = f"Table contains {len(df)} rows and {len(df.columns)} columns:\n" + "\n".join(columns_info)
        
        return df, auto_description, None
    except Exception as e:
        return None, "", str(e)

def get_sheet_names(file_path):
    """Get all sheet names from Excel file"""
    try:
        excel_file = pd.ExcelFile(file_path)
        return excel_file.sheet_names
    except:
        return []

def process_uploaded_file(file, sheet_name, custom_description):
    """Process uploaded Excel file"""
    global current_df, current_engine, current_table_description

    if file is None:
        return "Please upload an Excel file first.", "", gr.update(choices=["No sheets available"], value="No sheets available", visible=False), gr.update(visible=False)

    try:
        temp_file_created = False
        if hasattr(file, 'name'):
            tmp_file_path = file.name
        elif hasattr(file, 'read'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(file.read())
                tmp_file_path = tmp.name
            temp_file_created = True
        else:
            return gr.update(choices=["Unsupported file type"], value="Unsupported file type", visible=False)

        # Get sheet names
        sheet_names = get_sheet_names(tmp_file_path)
        if not sheet_names:
            if temp_file_created:
                os.unlink(tmp_file_path)
            return "Error: No sheets found in the Excel file.", "", gr.update(choices=["No sheets available"], value="No sheets available", visible=False), gr.update(visible=False)

        # Handle sheet selection
        if len(sheet_names) > 1:
            sheet_choices = ["Select Sheet"] + sheet_names
            if sheet_name and sheet_name in sheet_names:
                selected_sheet = sheet_name
                sheet_value = sheet_name
            else:
                selected_sheet = sheet_names[0]
                sheet_value = "Select Sheet"
            sheet_dropdown = gr.update(choices=sheet_choices, value=sheet_value, visible=True)
        else:
            selected_sheet = sheet_names[0]
            sheet_dropdown = gr.update(choices=[sheet_names[0]], value=sheet_names[0], visible=False)

        # Load data
        current_df, auto_description, error = load_excel_data(tmp_file_path, selected_sheet)
        if temp_file_created:
            os.unlink(tmp_file_path)

        if error:
            return f"Error loading file: {error}", "", sheet_dropdown, gr.update(visible=False)
        if current_df is None:
            return "Failed to load Excel file.", "", sheet_dropdown, gr.update(visible=False)

        # Create in-memory SQLite database
        current_engine = create_engine('sqlite:///:memory:')
        current_df.to_sql('data', con=current_engine, index=False, if_exists='replace')

        # Use custom description if provided, otherwise use auto-generated
        current_table_description = custom_description.strip() if custom_description.strip() else auto_description

        preview = current_df.head(5).to_html(index=False, classes="table table-striped")
        success_msg = f"✅ Successfully loaded Excel file!\n\n**Data Preview:**\n{preview}\n\n**Rows:** {len(current_df)} | **Columns:** {len(current_df.columns)}"

        return success_msg, auto_description, sheet_dropdown, gr.update(visible=True)

    except Exception as e:
        return f"Error processing file: {str(e)}", "", gr.update(choices=["No sheets available"], value="No sheets available", visible=False), gr.update(visible=False)
    

def text2SQL(user_query: str, table_description: str) -> str:
    """Generate SQL query from natural language"""
    query_template = """Given the following table description and a user query, generate a SQL query that retrieves the requested information from the table named 'data'.

Important guidelines:
- Only return the SQL query without any additional text or explanation
- Use proper SQL syntax for SQLite
- The table name is always 'data'
- Be careful with column names and use exact matches

Table Description: {table_description}
User Query: {query}

SQL Query:"""
    
    query_prompt_template = PromptTemplate(
        input_variables=["query", "table_description"],
        template=query_template
    )
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )
    chain = LLMChain(llm=llm, prompt=query_prompt_template)
    response = chain.invoke(input={"query": user_query, "table_description": table_description})
    return response['text'].strip()

def analyze_results(user_query, results, table_description):
    """Use LLM to generate analysis based on results"""
    if results.empty:
        return "No data matched your query, so no analysis can be provided."
    
    # Convert results to markdown table for context
    table_md = results.head(20).to_markdown(index=False)
    analysis_prompt = f"""Given the following user question, table description, and query results, provide a concise and insightful analysis.

User Question: {user_query}

Table Description: {table_description}

Query Results:
{table_md}

Please provide a clear, concise analysis that directly addresses the user's question based on the data shown above:"""
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke(analysis_prompt)
    return response.content.strip() if hasattr(response, "content") else str(response)

def chat_fn(message, history):
    """Main chat function"""
    global current_df, current_engine, current_table_description
    
    if current_df is None or current_engine is None:
        return history + [(message, "❌ Please upload an Excel file first before asking questions.")]
    
    if not message.strip():
        return history + [(message, "❌ Please enter a question.")]
    
    try:
        # Generate SQL from user query
        sql_query = text2SQL(user_query=message, table_description=current_table_description)
        sql_query_clean = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Execute SQL
        results = execute_query(sql_query_clean, current_engine)
        
        if results.empty:
            answer = "No results found for your query."
            analysis = "No data matched your query criteria."
        else:
            # Format results as HTML table
            answer = tabulate(results.head(50), headers='keys', tablefmt='html', showindex=False)
            analysis = analyze_results(message, results, current_table_description)
        
        # Format response
        response = f"""**🔍 Generated SQL Query:**
```sql
{sql_query_clean}
```

**📊 Results:**
{answer}

**💡 Analysis:**
{analysis}

**📈 Result Summary:** {len(results)} rows returned"""
        
        return history + [(message, response)]
        
    except Exception as e:
        if "no such table: data" in str(e):
            error_response = (
                "❌ **Error:** The table 'data' does not exist in the database.\n"
                "This usually means the Excel file was not loaded correctly or the table creation failed.\n\n"
                f"**Generated SQL:** ```sql\n{sql_query_clean}\n```\n\n"
                "Please re-upload your Excel file and try again."
         )
        else:
            error_response = (
                f"❌ **Error executing query:** {str(e)}\n\n"
                f"**Generated SQL:** ```sql\n{sql_query_clean}\n```\n\n"
                "Please try rephrasing your question or check the table structure."
         )
        return history + [(message, error_response)]

def clear_chat():
    """Clear chat history"""
    return []

def get_sample_questions():
    """Generate sample questions based on current data"""
    if current_df is None:
        return "Upload data first to see sample questions."
    
    samples = [
        "Show me the first 10 rows",
        "What are the column names and their data types?",
        "How many rows are in the table?",
        "Show me summary statistics",
        "What are the unique values in each column?"
    ]
    
    # Add column-specific questions
    if len(current_df.columns) > 0:
        first_col = current_df.columns[0]
        samples.append(f"Show me all unique values in {first_col}")
        
        # Add numeric column questions
        numeric_cols = current_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            samples.append(f"What's the average {numeric_cols[0]}?")
            samples.append(f"Show me the highest {numeric_cols[0]} values")
    
    return "**📝 Sample Questions:**\n" + "\n".join([f"• {q}" for q in samples])

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Excel SQL Chatbot") as demo:
    gr.Markdown("""
    # 📊 Excel SQL Chatbot
    
    Upload an Excel file and ask questions about your data in natural language!
    The system will convert your questions to SQL queries and provide insights.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # File upload section
            gr.Markdown("## 📁 Upload Excel File")
            file_input = gr.File(label="Choose Excel File", file_types=[".xlsx", ".xls"])
            sheet_dropdown = gr.Dropdown(label="Select Sheet", choices=["No sheets available"], value="No sheets available", visible=False)
            
            # Table description section
            gr.Markdown("## 📝 Table Description")
            gr.Markdown("*Describe your table structure (optional - auto-generated if empty)*")
            description_input = gr.Textbox(
                label="Custom Table Description",
                placeholder="e.g., The table contains student marks with columns: Name, Math, Science, English, Location.",
                lines=3,
                value=""
            )
            auto_description = gr.Textbox(
                label="Auto-Generated Description",
                lines=5,
                interactive=False,
                visible=False
            )
            
            process_btn = gr.Button("🔄 Process File", variant="primary")
            
        with gr.Column(scale=2):
            # Status and preview
            status_output = gr.Markdown("👆 Upload an Excel file to get started!")
            
            # Chat interface
            chatbot = gr.Chatbot(
                label="Chat with your data",
                height=400,
                visible=False
            )
            
            msg_input = gr.Textbox(
                label="Ask a question about your data",
                placeholder="e.g., What's the average score in Math?",
                visible=False
            )
            
            with gr.Row(visible=False) as button_row:
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")
                sample_btn = gr.Button("Show Sample Questions")
            
            sample_output = gr.Markdown("", visible=False)
    
    # Event handlers

    file_input.change(
        fn=lambda: gr.update(visible=False),
        outputs=chatbot
    )

    def handle_file_process(file, sheet_name, custom_description):
        """Handle file processing with proper sheet validation"""
        # If sheet_name is None or empty, pass None to avoid dropdown issues
        safe_sheet_name = sheet_name if sheet_name and sheet_name != "Select Sheet" else None
        return process_uploaded_file(file, safe_sheet_name, custom_description)

    # INSERTED FUNCTION
    def handle_sheet_change(file, sheet_name, custom_description):
        """Handle sheet selection change"""
        if not file:
            return "Please upload a file first.", "", gr.update(choices=["No sheets available"], value="No sheets available", visible=False), gr.update(visible=False)
        if not sheet_name or sheet_name == "Select Sheet" or sheet_name == "No sheets available":
            return "Please select a valid sheet.", "", gr.update(), gr.update(visible=False)
        return process_uploaded_file(file, sheet_name, custom_description)

    process_btn.click(
        fn=handle_file_process,
        inputs=[file_input, sheet_dropdown, description_input],
        outputs=[status_output, auto_description, sheet_dropdown, chatbot]
    ).then(
        fn=lambda: [gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)],
        outputs=[msg_input, button_row, sample_output, auto_description]
    )

    sheet_dropdown.change(
        fn=handle_sheet_change,
        inputs=[file_input, sheet_dropdown, description_input],
        outputs=[status_output, auto_description, sheet_dropdown, chatbot]
    )
    
    msg_input.submit(
        fn=chat_fn,
        inputs=[msg_input, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=msg_input
    )
    
    submit_btn.click(
        fn=chat_fn,
        inputs=[msg_input, chatbot],
        outputs=chatbot
    ).then(
        fn=lambda: "",
        outputs=msg_input
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=chatbot
    )
    
    sample_btn.click(
        fn=get_sample_questions,
        outputs=sample_output
    )
    
    # Add footer
    gr.Markdown("""
    ---
    **💡 Tips:**
    - Upload .xlsx or .xls files
    - Ask questions in natural language
    - The system auto-generates SQL queries
    - Get insights and analysis automatically
    """)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)