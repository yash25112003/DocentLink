"""
Agents package for AskMyProf application.
"""

from .email_agent_system import EmailAgentSystem
import logging

logging.basicConfig(level=logging.DEBUG)

__all__ = ['EmailAgentSystem'] 