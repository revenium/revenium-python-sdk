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
    ...     organization_name="AcmeCorp",
    ...     subscription_id="82764738",
    ...     product_name="Platinum"
    ... )
    >>> 
    >>> # Execute - all metadata automatically tracked
    >>> result = crew.kickoff()
"""

import uuid
import logging
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from ...._core.context import injected_metadata_scope
from ...._core.patch_registry import is_patched
from ....job_context import JobContext
from ..context import metadata_context

logger = logging.getLogger("revenium_middleware.crewai")

# CrewAI 1.x native LLM classes, keyed by class name so no CrewAI provider
# module (each needs its own optional vendor SDK) is imported to match them.
_NATIVE_LLM_MIDDLEWARE = {
    "OpenAICompletion": ("openai.resources.chat.completions.Completions.create", "revenium_middleware.openai"),
    "AnthropicCompletion": ("anthropic.resources.messages.messages.Messages.create", "revenium_middleware.anthropic"),
    "GeminiCompletion": ("google.genai.models.Models.generate_content", "revenium_middleware.google.google_ai"),
}
_UNMETERED_NATIVE_LLMS = frozenset({"AzureCompletion", "BedrockCompletion", "SnowflakeCompletion"})
_warned_llm_classes = set()


def _native_llm_class_name(llm: Any) -> Optional[str]:
    for klass in type(llm).__mro__:
        if klass.__name__ in _NATIVE_LLM_MIDDLEWARE or klass.__name__ in _UNMETERED_NATIVE_LLMS:
            return klass.__name__
    return None


def _warn_if_unmetered(llm: Any) -> None:
    class_name = _native_llm_class_name(llm)
    if class_name is None or class_name in _warned_llm_classes:
        return
    if class_name in _UNMETERED_NATIVE_LLMS:
        _warned_llm_classes.add(class_name)
        logger.warning(
            "CrewAI %s calls are not metered by Revenium; build that LLM with is_litellm=True "
            "and import revenium_middleware.litellm.client.middleware",
            type(llm).__name__,
        )
        return
    patch_key, middleware_module = _NATIVE_LLM_MIDDLEWARE[class_name]
    if not is_patched(patch_key):
        _warned_llm_classes.add(class_name)
        logger.warning("CrewAI %s calls are not metered: import %s before kickoff", type(llm).__name__, middleware_module)


def _is_crew_streaming_output(result: Any) -> bool:
    return any(klass.__name__ == "CrewStreamingOutput" for klass in type(result).__mro__)


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
    - Agent role and task type for each task execution (CrewAI before 0.203 only)
    - Optional agentic job fields (agentic_job_id/name/type/version) for
      Revenium's agentic outcomes API

    All metadata is automatically injected into every model call the crew
    makes through a provider whose Revenium middleware is imported (LiteLLM,
    OpenAI, Anthropic), eliminating the need for manual monkey-patching.

    When constructed with an ``agentic_job_id``, the wrapper can also report
    or amend the job's business outcome via ``report_job_outcome`` and
    ``amend_job_outcome`` (requires a write-scope ``rev_sk_`` API key).

    Attributes:
        organization_name: Customer or department name
        subscription_id: Billing plan reference
        product_name: Product or feature name
        organization_id: Deprecated, use organization_name
        product_id: Deprecated, use product_name
        trace_id: Unique identifier for this crew execution (auto-generated)
        agentic_job_id: Optional agentic job identifier for outcome reporting
        agentic_job_name: Optional human-readable agentic job name
        agentic_job_type: Optional agentic job type/category
        agentic_job_version: Optional agentic job version
        team_id: Optional Revenium team ID used for outcome reporting
    """

    def __init__(
        self,
        agents: List[Any],
        tasks: List[Any],
        organization_id: Optional[str] = None,
        subscription_id: Optional[str] = None,
        product_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        agentic_job_id: Optional[str] = None,
        agentic_job_name: Optional[str] = None,
        agentic_job_type: Optional[str] = None,
        agentic_job_version: Optional[str] = None,
        team_id: Optional[str] = None,
        process: Optional[Any] = None,
        verbose: bool = False,
        organization_name: Optional[str] = None,
        product_name: Optional[str] = None,
        **crew_kwargs
    ):
        """
        Initialize the Revenium-wrapped Crew.

        Args:
            agents: List of CrewAI Agent objects
            tasks: List of CrewAI Task objects
            organization_id: Deprecated, use organization_name.
            subscription_id: Reference to a billing plan in non-Revenium systems
            product_id: Deprecated, use product_name.
            trace_id: Optional unique identifier for this execution. If not provided,
                     a UUID will be generated automatically.
            agentic_job_id: Optional agentic job identifier. When set, it is
                     injected into metering metadata and enables
                     report_job_outcome/amend_job_outcome.
            agentic_job_name: Optional human-readable name for the agentic job.
            agentic_job_type: Optional agentic job type/category (e.g.
                     "customer_support").
            agentic_job_version: Optional version string for the agentic job.
            team_id: Optional Revenium team ID used when reporting or amending
                     the job outcome (falls back to REVENIUM_TEAM_ID).
            process: Optional CrewAI process type (sequential, hierarchical, etc.)
            verbose: Whether to enable verbose logging
            organization_name: Customer or department name from non-Revenium systems
            product_name: Your product or feature making the AI call
            **crew_kwargs: Additional arguments to pass to Crew constructor

        Example:
            >>> crew = ReveniumCrewWrapper(
            ...     agents=[agent1, agent2],
            ...     tasks=[task1, task2],
            ...     organization_name="AcmeCorp",
            ...     subscription_id="82764738",
            ...     product_name="Platinum",
            ...     agentic_job_id="support-456",
            ...     agentic_job_type="customer_support",
            ...     agentic_job_version="2.0",
            ...     verbose=True
            ... )
            >>> result = crew.kickoff()
            >>> crew.report_job_outcome(
            ...     execution_status="SUCCESS",
            ...     outcome_type="DEFLECTED",
            ...     outcome_value=25.0,
            ... )
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. Install it with the extra for your agents' models, e.g.: "
                "pip install 'revenium-python-sdk[openai]' crewai"
            )
        
        self.subscription_id = subscription_id
        self.organization_name = organization_name if organization_name is not None else organization_id
        self.product_name = product_name if product_name is not None else product_id
        self.trace_id = trace_id or str(uuid.uuid4())
        self.agentic_job_id = agentic_job_id
        self.agentic_job_name = agentic_job_name
        self.agentic_job_type = agentic_job_type
        self.agentic_job_version = agentic_job_version
        self.team_id = team_id

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

    @property
    def organization_id(self) -> Optional[str]:
        return self.organization_name

    @organization_id.setter
    def organization_id(self, value: Optional[str]) -> None:
        self.organization_name = value

    @property
    def product_id(self) -> Optional[str]:
        return self.product_name

    @product_id.setter
    def product_id(self, value: Optional[str]) -> None:
        self.product_name = value

    def _execution_metadata(self) -> Dict[str, Any]:
        """Build the metadata dict injected for every crew execution."""
        optional = {
            "organization_name": self.organization_name,
            "subscription_id": self.subscription_id,
            "product_name": self.product_name,
            "agentic_job_id": self.agentic_job_id,
            "agentic_job_name": self.agentic_job_name,
            "agentic_job_type": self.agentic_job_type,
            "agentic_job_version": self.agentic_job_version,
        }
        metadata: Dict[str, Any] = {"trace_id": self.trace_id}
        metadata.update((field, value) for field, value in optional.items() if value is not None)
        return metadata

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None):
        """
        Execute the crew with Revenium metadata tracking.

        Args:
            inputs: Optional inputs to pass to the crew

        Returns:
            The result from the crew execution
        """
        self._warn_unmetered_llms()
        with self._attribution_scope():
            logger.debug(f"Starting crew execution with trace_id: {self.trace_id}")

            # Setup task-level metadata (version-aware: monkey-patching or callbacks)
            self._setup_task_callbacks()

            try:
                return self._attribute_stream(self._crew.kickoff(inputs=inputs))
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
        self._warn_unmetered_llms()
        with self._attribution_scope():
            logger.debug(f"Starting async crew execution with trace_id: {self.trace_id}")

            # Setup task-level metadata (version-aware: monkey-patching or callbacks)
            self._setup_task_callbacks()

            try:
                if hasattr(self._crew, 'kickoff_async'):
                    return self._attribute_stream(await self._crew.kickoff_async(inputs=inputs))
                else:
                    # Fallback to sync if async not available
                    return self._attribute_stream(self._crew.kickoff(inputs=inputs))
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
        self._warn_unmetered_llms()
        with self._attribution_scope():
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

    @contextmanager
    def _attribution_scope(self):
        """Attribute every model call in the enclosed code to this crew.

        The LiteLLM client wrapper reads ``metadata_context``; the OpenAI and
        Anthropic middlewares read the core injected metadata. Both get the
        same fields and both are restored on exit.
        """
        metadata = self._execution_metadata()
        with metadata_context.set(**metadata), injected_metadata_scope(metadata):
            yield

    def _warn_unmetered_llms(self) -> None:
        """Warn once per CrewAI LLM class whose calls no imported middleware meters."""
        try:
            llms = [getattr(self._crew, "manager_llm", None), getattr(self._crew, "planning_llm", None)]
            for agent in self._agents:
                llms.append(getattr(agent, "llm", None))
                llms.append(getattr(agent, "function_calling_llm", None))
            for llm in llms:
                if llm is not None:
                    _warn_if_unmetered(llm)
        except Exception as e:
            logger.debug("Could not check the crew's LLMs for missing middleware: %s", e)

    def _attribute_stream(self, result: Any) -> Any:
        """Attribute a streaming crew's model calls, which run while it is iterated."""
        if not _is_crew_streaming_output(result):
            return result
        if getattr(result, "_sync_iterator", None) is not None:
            result._sync_iterator = self._scoped_steps(result._sync_iterator)
        if getattr(result, "_async_iterator", None) is not None:
            result._async_iterator = self._scoped_async_steps(result._async_iterator)
        return result

    def _scoped_steps(self, iterator):
        # Each step gets its own scope: a scope held across a yield would be
        # reset from whatever context resumes the generator.
        try:
            while True:
                with self._attribution_scope():
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        return
                yield chunk
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    async def _scoped_async_steps(self, iterator):
        try:
            while True:
                with self._attribution_scope():
                    try:
                        chunk = await iterator.__anext__()
                    except StopAsyncIteration:
                        return
                yield chunk
        finally:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                await aclose()

    def report_job_outcome(
        self,
        execution_status: str,
        *,
        team_id: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[Any] = None,
        retry_attempts: Optional[int] = None,
        retry_initial_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
        **outcome_kwargs: Any,
    ) -> None:
        """Report the business outcome for this crew's agentic job.

        Requires the wrapper to have been constructed with agentic_job_id, and
        a write-scope API key (rev_sk_) — see JobContext.report_outcome.

        The retry knobs govern the outcome POST's transient-error retry; omit
        them for the default bounded schedule.
        """
        handle = self._job_handle(
            team_id=team_id, api_key=api_key, http_client=http_client,
            retry_attempts=retry_attempts,
            retry_initial_seconds=retry_initial_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        try:
            handle.report_outcome(execution_status, **outcome_kwargs)
        finally:
            handle.close()

    def amend_job_outcome(
        self,
        reason: str,
        *,
        team_id: Optional[str] = None,
        api_key: Optional[str] = None,
        http_client: Optional[Any] = None,
        retry_attempts: Optional[int] = None,
        retry_initial_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
        **amend_kwargs: Any,
    ) -> Dict[str, Any]:
        """Amend a previously-reported outcome for this crew's agentic job.

        The retry knobs govern the amendment PATCH's retry (429 only, since
        PATCH-amend is not idempotent); omit them for the default schedule.
        """
        handle = self._job_handle(
            team_id=team_id, api_key=api_key, http_client=http_client,
            retry_attempts=retry_attempts,
            retry_initial_seconds=retry_initial_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        try:
            return handle.amend_outcome(reason, **amend_kwargs)
        finally:
            handle.close()

    def _job_handle(
        self,
        *,
        team_id: Optional[str],
        api_key: Optional[str],
        http_client: Optional[Any],
        retry_attempts: Optional[int] = None,
        retry_initial_seconds: Optional[float] = None,
        retry_max_seconds: Optional[float] = None,
    ) -> JobContext:
        if not self.agentic_job_id:
            raise ValueError(
                "report_job_outcome/amend_job_outcome require agentic_job_id "
                "on the ReveniumCrewWrapper constructor"
            )
        return JobContext.attach(
            job_id=self.agentic_job_id,
            team_id=team_id or self.team_id,
            api_key=api_key,
            http_client=http_client,
            retry_attempts=retry_attempts,
            retry_initial_seconds=retry_initial_seconds,
            retry_max_seconds=retry_max_seconds,
        )

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
        Log each task's agent and type (CrewAI 0.203.0+).

        Task callbacks run after the task, so they cannot inject metadata into
        model calls; the crew-level attribution scope does that. The task's own
        ``callback`` is left untouched, so a wrapper reused across runs never
        stacks a layer on it.
        """
        for task in self._tasks:
            agent_role = self._get_agent_role(task)
            task_type = self._get_task_type(task)
            logger.debug(f"Tracking task with agent={agent_role}, task_type={task_type}")

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

