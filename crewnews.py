import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
 
# Your tool definition
search_tool = SerperDevTool()
 
# News collector agent
news_collector = Agent(
    role='Technology News Harvester',
    goal='Collect daily technology news from various sources on a specific topic.',
    tools=[search_tool],
    verbose=True,
    memory=False,
    cache=False,
    backstory=(
        "Dedicated to scouring the internet for the latest tech news, "
        "you aim to gather the most relevant articles to keep the tech community well-informed."
    ),
    allow_delegation=False
)
 
# Task for fetching news
fetch_news_task = Task(
    description=(
        "Search for the latest technology articles related to the chosen topic from the past 24 hours. "
        "Ensure sources are credible and provide recent updates."
    ),
    expected_output='Raw data with news headline, date, source, article URL and summary ready for post-processing.',
    agent=news_collector
)
 
# Data organizer agent
content_organizer = Agent(
    role='Data Organizer',
    goal='Convert raw news data into a structured format for reporting.',
    backstory=(
        "Expert in data manipulation, you transform unstructured data into organized, concise formats. "
        "Your work ensures that the information is not only accurate but also pleasing to read."
    ),
    allow_delegation=False,
    verbose=True,
    memory=False,
    cache=False
)
 
# Task for organizing data
organize_data_task = Task(
    description=(
        "Take raw data from the news collector and organize it into a structured format. "
        "Extract and highlight headlines, publication dates, article URLs and summaries."
    ),
    expected_output='Organized content ready for Markdown formatting.',
    agent=content_organizer
)
 
# Report compiler agent
markdown_generator = Agent(
    role='Report Compiler',
    goal='Generate a Markdown document from structured news data.',
    backstory=(
        "As a compiler of information, your role is to take organized data and weave it into a coherent, "
        "engaging narrative in Markdown format."
    ),
    allow_delegation=False,
    verbose=True,
    memory=False,
    cache=False
)
 
# Task for generating Markdown
generate_markdown_task = Task(
    description='Compile the structured data into a Markdown formatted document.',
    expected_output='A well-formatted Markdown document containing the latest technology news.',
    agent=markdown_generator
)
 
# Initialize the Crew with defined agents and tasks
tech_news_crew = Crew(
    name='Daily Tech News Crew',
    agents=[news_collector, content_organizer, markdown_generator],
    tasks=[fetch_news_task, organize_data_task, generate_markdown_task],
    process=Process.sequential,  # Ensures tasks are executed in the order they are added
    verbose=True,
    memory=False,
    cache=False,
    max_rpm=100,
    share_crew=True
)
 
# Entry point to execute the crew
if __name__ == '__main__':
    # Execute the crew with an input topic for news collection
    # Set OpenAPI Key before: export OPENAI_API_KEY="your_openai_api_key_here"
    result = tech_news_crew.kickoff(inputs={'topic': 'software engineering'})
    print("Crew Execution Result:", result)
    
    # Write the result to a Markdown file
    with open("news.md", "w") as file:
        file.write(str(result))