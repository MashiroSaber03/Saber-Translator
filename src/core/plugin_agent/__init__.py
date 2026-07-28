from .controller import PluginAgentController
from .models import (
    LockedPluginTarget,
    PluginAgentEvent,
    PluginAgentMessage,
    PluginAgentSession,
    PluginTargetProposal,
)
from .skill import (
    PLUGIN_AGENT_OVERVIEW,
    PLUGIN_AGENT_PROMPT_EXAMPLES,
    get_plugin_agent_settings_payload,
    get_plugin_builder_skill_markdown,
)


def __getattr__(name: str):
    """Keep legacy execution helpers lazy so v2 API stays metadata-only."""

    if name == "PluginAgentRuntime":
        from .runtime import PluginAgentRuntime

        return PluginAgentRuntime
    if name == "PluginAgentToolExecutor":
        from .tools import PluginAgentToolExecutor

        return PluginAgentToolExecutor
    raise AttributeError(name)

__all__ = [
    "LockedPluginTarget",
    "PluginAgentController",
    "PluginAgentEvent",
    "PluginAgentMessage",
    "PluginAgentRuntime",
    "PluginAgentSession",
    "PluginAgentToolExecutor",
    "PluginTargetProposal",
    "PLUGIN_AGENT_OVERVIEW",
    "PLUGIN_AGENT_PROMPT_EXAMPLES",
    "get_plugin_agent_settings_payload",
    "get_plugin_builder_skill_markdown",
]
