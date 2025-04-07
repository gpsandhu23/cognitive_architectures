"""Default prompts used by the agent."""

JOKE_SETUP_PROMPT = """Your job is to generate a joke setup, just the setup, no punchline.

System time: {system_time}"""

JOKE_PUNCHLINE_PROMPT = """Your job is to generate a joke punchline for the setup from the previous message, just the punchline, no setup.

System time: {system_time}"""
