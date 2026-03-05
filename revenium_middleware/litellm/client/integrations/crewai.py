"""
CrewAI integration for Revenium LiteLLM middleware.

This module provides a simple wrapper for CrewAI that automatically tracks
agent and task metadata without requiring manual monkey-patching.

Example:
    >>> from revenium_middleware.litellm.client.integrations.crewai import ReveniumCrewWrapper
    >>> from crewai import Agent, Task, Crew
    >>> 
    >>> # Create your agents and tasks as normal
    >>> agent = Agent(role="Lead Analyst", ...)
    >>> task = Task(description="Research...", agent=agent)
    >>> 
    >>> # Wrap with Revenium tracking
    >>> crew = ReveniumCrewWrapper(
    ...     agents=[agent],
    ...     tasks=[task],
    ...     organization_id="AcmeCorp",
    ...     subscription_id="82764738",
    ...     product_id="Platinum"
    ... )
    >>> 
    >>> # Execute - all metadata automatically tracked
    >>> result = crew.kickoff()
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from ..context import metadata_context

logger = logging.getLogger("revenium_middleware.crewai")

try:
    from crewai import Crew, Agent, Task
    from crewai.llm import LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Crew = object  # type: ignore
    Agent = object  # type: ignore
    Task = object  # type: ignore
    LLM = object  # type: ignore


def _get_crewai_version() -> tuple:
    """
    Get the installed CrewAI version as a tuple of integers.

    Returns:
        Tuple of (major, minor, patch) version numbers
        Returns (0, 0, 0) if version cannot be determined
    """
    try:
        import crewai
        if hasattr(crewai, '__version__'):
            version_str = crewai.__version__
            # Parse version string like "0.203.0" into (0, 203, 0)
            parts = version_str.split('.')
            return tuple(int(p) for p in parts[:3])
    except Exception as e:
        logger.warning(f"Could not determine CrewAI version: {e}")

    return (0, 0, 0)


def _supports_task_monkey_patching() -> bool:
    """
    Check if the installed CrewAI version supports monkey-patching task execution.

    CrewAI < 0.203.0 allows monkey-patching task.execute_sync
    CrewAI >= 0.203.0 uses Pydantic models that prevent monkey-patching

    Returns:
        True if monkey-patching is supported, False otherwise
    """
    version = _get_crewai_version()

    # Version (0, 0, 0) means we couldn't determine version - try monkey-patching
    if version == (0, 0, 0):
        logger.info("CrewAI version unknown - will attempt monkey-patching with fallback")
        return True

    # CrewAI < 0.203.0 supports monkey-patching
    if version < (0, 203, 0):
        logger.info(f"CrewAI {'.'.join(map(str, version))} detected - using monkey-patching for task-level metadata")
        return True

    # CrewAI >= 0.203.0 uses Pydantic models
    logger.info(f"CrewAI {'.'.join(map(str, version))} detected - using callback approach (crew-level metadata only)")
    return False


class ReveniumCrewWrapper:
    """
    Wrapper for CrewAI Crew that automatically injects Revenium metadata.
    
    This class wraps a CrewAI Crew and automatically tracks:
    - Organization, subscription, and product IDs
    - Unique trace_id for each crew execution
    - Agent role for each agent interaction
    - Task type for each task execution
    
    All metadata is automatically injected into LiteLLM calls via the
    context API, eliminating the need for manual monkey-patching.
    
    Attributes:
        organization_id: Customer or department ID
        subscription_id: Billing plan reference
        product_id: Product or feature identifier
        trace_id: Unique identifier for this crew execution (auto-generated)
    """
    
    def __init__(
        self,
        agents: List[Any],
        tasks: List[Any],
        organization_id: str,
        subscription_id: str,
        product_id: str,
        trace_id: Optional[str] = None,
        process: Optional[Any] = None,
        verbose: bool = False,
        **crew_kwargs
    ):
        """
        Initialize the Revenium-wrapped Crew.
        
        Args:
            agents: List of CrewAI Agent objects
            tasks: List of CrewAI Task objects
            organization_id: Customer or department ID from non-Revenium systems
            subscription_id: Reference to a billing plan in non-Revenium systems
            product_id: Your product or feature making the AI call
            trace_id: Optional unique identifier for this execution. If not provided,
                     a UUID will be generated automatically.
            process: Optional CrewAI process type (sequential, hierarchical, etc.)
            verbose: Whether to enable verbose logging
            **crew_kwargs: Additional arguments to pass to Crew constructor
        
        Example:
            >>> crew = ReveniumCrewWrapper(
            ...     agents=[agent1, agent2],
            ...     tasks=[task1, task2],
            ...     organization_id="AcmeCorp",
            ...     subscription_id="82764738",
            ...     product_id="Platinum",
            ...     verbose=True
            ... )
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. Install it with: "
                "pip install 'revenium-python-sdk[litellm]'"
            )
        
        self.organization_id = organization_id
        self.subscription_id = subscription_id
        self.product_id = product_id
        self.trace_id = trace_id or str(uuid.uuid4())
        
        # Store agents and tasks for metadata extraction
        self._agents = agents
        self._tasks = tasks
        
        # Create the underlying Crew
        crew_args = {
            'agents': agents,
            'tasks': tasks,
            'verbose': verbose,
            **crew_kwargs
        }
        
        if process is not None:
            crew_args['process'] = process
        
        self._crew = Crew(**crew_args)

        logger.info(
            f"Initialized Revenium Crew wrapper with trace_id: {self.trace_id}"
        )

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None):
        """
        Execute the crew with Revenium metadata tracking.

        Args:
            inputs: Optional inputs to pass to the crew

        Returns:
            The result from the crew execution
        """
        # Set base metadata for the entire crew execution
        # This will be picked up by the middleware for all LiteLLM calls
        with metadata_context.set(
            organization_id=self.organization_id,
            subscription_id=self.subscription_id,
            product_id=self.product_id,
            trace_id=self.trace_id
        ):
            logger.debug(f"Starting crew execution with trace_id: {self.trace_id}")

            # Setup task-level metadata (version-aware: monkey-patching or callbacks)
            self._setup_task_callbacks()

            try:
                return self._crew.kickoff(inputs=inputs)
            finally:
                # Clean up any monkey-patches
                self._unpatch_task_execution()

    async def kickoff_async(self, inputs: Optional[Dict[str, Any]] = None):
        """
        Execute the crew asynchronously with Revenium metadata tracking.

        Args:
            inputs: Optional inputs to pass to the crew

        Returns:
            The result from the crew execution
        """
        # Set base metadata for the entire crew execution
        with metadata_context.set(
            organization_id=self.organization_id,
            subscription_id=self.subscription_id,
            product_id=self.product_id,
            trace_id=self.trace_id
        ):
            logger.debug(f"Starting async crew execution with trace_id: {self.trace_id}")

            # Setup task-level metadata (version-aware: monkey-patching or callbacks)
            self._setup_task_callbacks()

            try:
                if hasattr(self._crew, 'kickoff_async'):
                    return await self._crew.kickoff_async(inputs=inputs)
                else:
                    # Fallback to sync if async not available
                    return self._crew.kickoff(inputs=inputs)
            finally:
                # Clean up any monkey-patches
                self._unpatch_task_execution()

    def train(self, n_iterations: int, inputs: Optional[Dict[str, Any]] = None):
        """
        Train the crew with Revenium metadata tracking.

        Args:
            n_iterations: Number of training iterations
            inputs: Optional inputs to pass to the crew

        Returns:
            The result from the training
        """
        # Set base metadata for the entire training session
        with metadata_context.set(
            organization_id=self.organization_id,
            subscription_id=self.subscription_id,
            product_id=self.product_id,
            trace_id=self.trace_id
        ):
            logger.debug(f"Starting crew training with trace_id: {self.trace_id}")

            # Setup task-level metadata (version-aware: monkey-patching or callbacks)
            self._setup_task_callbacks()

            try:
                if hasattr(self._crew, 'train'):
                    return self._crew.train(n_iterations=n_iterations, inputs=inputs)
                else:
                    raise AttributeError("Crew does not support training")
            finally:
                # Clean up any monkey-patches
                self._unpatch_task_execution()

    def _setup_task_callbacks(self):
        """
        Setup task execution tracking with version-aware approach.

        For CrewAI < 0.203.0: Uses monkey-patching to inject metadata during execution
        For CrewAI >= 0.203.0: Uses callbacks (crew-level metadata only)
        """
        if _supports_task_monkey_patching():
            # Try monkey-patching approach (works with older CrewAI versions)
            self._patch_task_execution()
        else:
            # Fall back to callback approach (CrewAI 0.203.0+)
            self._setup_task_callbacks_only()

    def _patch_task_execution(self):
        """
        Patch task execution methods to inject agent and task metadata.

        This approach works with CrewAI < 0.203.0 where task.execute_sync can be
        monkey-patched. Provides full task-level metadata injection.
        """
        self._original_task_executes = {}

        for task in self._tasks:
            # Try to patch execute_sync if it exists
            try:
                if hasattr(task, 'execute_sync'):
                    original_execute = task.execute_sync
                    self._original_task_executes[id(task)] = original_execute

                    # Create wrapped version that accepts all arguments
                    def wrapped_execute(*args, original=original_execute, task_obj=task, **kwargs):
                        # Extract agent role and task description
                        agent_role = self._get_agent_role(task_obj)
                        task_type = self._get_task_type(task_obj)

                        # UPDATE metadata context (don't replace it!)
                        metadata_context.update(
                            agent=agent_role,
                            task_type=task_type
                        )
                        logger.debug(
                            f"Executing task with agent={agent_role}, "
                            f"task_type={task_type}"
                        )
                        return original(*args, **kwargs)

                    # Attempt to set the wrapped method
                    # Use object.__setattr__ to bypass Pydantic validation
                    try:
                        object.__setattr__(task, 'execute_sync', wrapped_execute)
                        logger.debug("Successfully patched task execution for task-level metadata")
                    except (ValueError, AttributeError, TypeError):
                        # If object.__setattr__ fails, try regular assignment
                        task.execute_sync = wrapped_execute
                        logger.debug("Successfully patched task execution using regular assignment")

            except (ValueError, AttributeError, TypeError) as e:
                # Pydantic validation error or other issue - fall back to callbacks
                logger.warning(
                    f"Could not monkey-patch task execution (likely CrewAI 0.203.0+): {e}. "
                    f"Falling back to callback approach (crew-level metadata only)"
                )
                # Clean up any partial patches
                self._original_task_executes.clear()
                # Use callback approach instead
                self._setup_task_callbacks_only()
                return

    def _unpatch_task_execution(self):
        """Restore original task execution methods."""
        if not hasattr(self, '_original_task_executes'):
            return

        for task in self._tasks:
            task_id = id(task)
            if task_id in self._original_task_executes:
                try:
                    # Use object.__setattr__ to bypass Pydantic validation
                    object.__setattr__(task, 'execute_sync', self._original_task_executes[task_id])
                except (ValueError, AttributeError, TypeError):
                    # If object.__setattr__ fails, try regular assignment
                    try:
                        task.execute_sync = self._original_task_executes[task_id]
                    except (ValueError, AttributeError, TypeError):
                        # Ignore errors during unpatch
                        pass

        self._original_task_executes.clear()

    def _setup_task_callbacks_only(self):
        """
        Setup callbacks for tasks (callback-only approach for CrewAI 0.203.0+).

        Note: Callbacks execute AFTER task completion, so they cannot inject
        metadata into LiteLLM calls. This approach only provides logging.
        """
        try:
            from crewai.tasks.task_output import TaskOutput
        except ImportError:
            logger.warning("Could not import TaskOutput - callbacks not available")
            return

        for task in self._tasks:
            # Extract agent role and task type for this task
            agent_role = self._get_agent_role(task)
            task_type = self._get_task_type(task)

            # Create a callback for logging only
            def task_callback(output: TaskOutput, agent=agent_role, task_t=task_type):
                """Callback executed after task completion (logging only)."""
                logger.debug(
                    f"Task completed: agent={agent}, task_type={task_t}"
                )

            # Store original callback if it exists
            original_callback = getattr(task, 'callback', None)

            # Create a combined callback that calls both
            if original_callback:
                def combined_callback(output: TaskOutput, orig=original_callback, new=task_callback):
                    new(output)
                    orig(output)
                task.callback = combined_callback
            else:
                task.callback = task_callback

            logger.debug(
                f"Setup callback for task with agent={agent_role}, task_type={task_type}"
            )
    
    def _get_agent_role(self, task: Any) -> str:
        """
        Extract agent role from a task.
        
        Args:
            task: CrewAI Task object
        
        Returns:
            Agent role string
        """
        if hasattr(task, 'agent') and task.agent and hasattr(task.agent, 'role'):
            return str(task.agent.role)
        return "unknown_agent"
    
    def _get_task_type(self, task: Any) -> str:
        """
        Extract task type from a task.
        
        Args:
            task: CrewAI Task object
        
        Returns:
            Task type string (derived from description or name)
        """
        # Try to get from task description or name
        if hasattr(task, 'description') and task.description:
            # Use first few words of description as task type
            desc = str(task.description).lower()
            # Extract first meaningful phrase (up to 30 chars)
            task_type = desc[:30].split('.')[0].strip()
            # Clean up for use as identifier
            task_type = task_type.replace(' ', '_')
            return task_type
        
        return "unknown_task"


__all__ = ['ReveniumCrewWrapper', 'CREWAI_AVAILABLE']

