"""
Simple CrewAI example with Revenium middleware.

This example shows two approaches for tracking metadata:
1. ReveniumCrewWrapper - Simple wrapper for static metadata
2. Decorators - For dynamic metadata from function arguments

Requirements:
    pip install revenium-python-sdk[litellm]

Environment Variables (.env file):
    OPENAI_API_KEY=your_openai_api_key
    REVENIUM_METERING_API_KEY=your_revenium_api_key
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Import middleware FIRST to enable patching
import revenium_middleware.litellm.client.middleware

from crewai import Agent, Task, Crew
from revenium_middleware.litellm.client import track_agent, track_task
from revenium_middleware.litellm.client.integrations.crewai import ReveniumCrewWrapper

# Verify environment variables
if not os.getenv("OPENAI_API_KEY") or not os.getenv("REVENIUM_METERING_API_KEY"):
    print("Error: OPENAI_API_KEY and REVENIUM_METERING_API_KEY required in .env file")
    exit(1)

# ============================================================================
# APPROACH 1: ReveniumCrewWrapper (Recommended for static metadata)
# ============================================================================

print("Example 1: Using ReveniumCrewWrapper\n")

researcher = Agent(
    role="Research Analyst",
    goal="Analyze market trends",
    backstory="Expert in market research",
    verbose=False
)

research_task = Task(
    description="Research the latest AI technology trends",
    agent=researcher,
    expected_output="Brief summary of AI trends"
)

# Wrap crew with Revenium tracking
crew = ReveniumCrewWrapper(
    agents=[researcher],
    tasks=[research_task],
    organization_name="AcmeCorp",
    subscription_id="premium-plan",
    product_name="ai-research-tool",
    verbose=False
)

result = crew.kickoff()
print(f"Result: {result}\n")

# ============================================================================
# APPROACH 2: Decorators (For dynamic metadata)
# ============================================================================

print("Example 2: Using Decorators for Dynamic Metadata\n")

@track_agent(name_from_arg="agent_name")
@track_task(type_from_arg="task_type")
def run_analysis(agent_name: str, task_type: str, query: str):
    """Run analysis with metadata from function arguments."""

    analyst = Agent(
        role=agent_name,
        goal=f"Perform {task_type}",
        backstory=f"Expert in {task_type}",
        verbose=False
    )

    task = Task(
        description=query,
        agent=analyst,
        expected_output="Analysis results"
    )

    crew = Crew(agents=[analyst], tasks=[task], verbose=False)
    return crew.kickoff()

result = run_analysis(
    agent_name="Market Analyst",
    task_type="competitive_analysis",
    query="Analyze top 3 AI competitors"
)

print(f"Result: {result}\n")
print("✅ Both examples completed successfully")

