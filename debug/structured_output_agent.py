r"""Demo: Using ``create_agent`` with structured output.

This script shows the minimal moving pieces needed to get a
schema-validated response out of a LangChain agent:

1. Define a Pydantic model describing the desired output shape.
2. Give the agent a tool it can call to gather information.
3. Create the agent with ``response_format`` set to the schema.
4. Invoke the agent and read the parsed result from
   ``result["structured_response"]``.
"""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from pydantic import BaseModel, Field
from zenpyre.utils.rich import configure_rich_logging, print_pretty

logger: logging.Logger = logging.getLogger(__name__)

# NOTE: verify this against the models actually available in your Ollama
# install (e.g. `ollama list`). "gemma4" / "-mlx" does not match any known
# released Ollama tag as of this writing.
MODEL = "ollama:gemma4:e2b-mlx"


# 1. Define the structured output schema
class WeatherReport(BaseModel):
    """Structured weather report extracted from the agent's response.

    Attributes:
        city: Name of the city the report is about.
        temperature_celsius: Temperature in Celsius.
        condition: Short description of the weather condition, e.g.
            "light rain" or "sunny".
        recommendation: A brief recommendation based on the weather,
            e.g. "bring an umbrella".
    """

    city: str = Field(description="Name of the city")
    temperature_celsius: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Short description of weather condition")
    recommendation: str = Field(description="A brief recommendation based on the weather")


# 2. (Optional) Define a tool the agent can call
def get_weather(city: str) -> str:
    """Look up the current weather for a given city.

    This is a dummy implementation for demo purposes: it does an exact,
    case-sensitive lookup against a small hardcoded table and falls
    back to a generic default for any city not in the table (including
    known cities spelled with different casing, e.g. ``"vancouver"``).

    Args:
        city: The name of the city to look up, e.g. ``"Vancouver"``.

    Returns:
        A short human-readable weather string, e.g. ``"14C, light
        rain"``. Returns ``"18C, clear skies"`` if ``city`` is not
        found in the lookup table.
    """
    fake_data = {
        "Vancouver": "14C, light rain",
        "Tokyo": "28C, sunny",
        "Paris": "20C, cloudy",
    }
    return fake_data.get(city, "18C, clear skies")


def main() -> None:
    """Run the structured-output agent demo end to end.

    Creates an agent with a weather-lookup tool and a ``WeatherReport``
    response format, invokes it with a sample user question, then logs
    both the raw message trace and the parsed structured output.
    """
    # 3. Create the agent with a structured output format
    agent = create_agent(
        model=MODEL,
        tools=[get_weather],
        response_format=WeatherReport,
        system_prompt=(
            "You are a helpful weather assistant. Use the get_weather tool "
            "to look up conditions, then summarize them clearly."
        ),
    )

    # 4. Invoke the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather like in Vancouver?"}]}
    )
    print_pretty(result)

    # 5. Extract the structured output
    structured: WeatherReport = result["structured_response"]

    logger.info("Raw messages:")
    for m in result["messages"]:
        logger.info("  %s: %s", m.type, m.content)

    logger.info("Structured output:")
    logger.info("  City: %s", structured.city)
    logger.info("  Temperature: %s\u00b0C", structured.temperature_celsius)
    logger.info("  Condition: %s", structured.condition)
    logger.info("  Recommendation: %s", structured.recommendation)


if __name__ == "__main__":
    configure_rich_logging(level=logging.INFO, show_path=False)
    main()
