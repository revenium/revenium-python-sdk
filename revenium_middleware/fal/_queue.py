"""One metering record per fal queue job, however its result is fetched.

Every way of reaching a queued result (``subscribe``, ``submit(...).get()``,
``result(application, request_id)``) ends in the request handle's ``get``, so
that is the only place a queue job is metered. The calls that start a job
record its context here under the fal request id, and ``get`` claims it.

A job this process did not submit has no caller context to keep, so only its
application is remembered (from ``get_handle``); whether it is metered is
decided by the scope that fetches its result.
"""

import threading
from collections import OrderedDict
from typing import Optional
from urllib.parse import urlparse

from ._call import UNKNOWN_APPLICATION, FalCall

# A job is tracked from submit until its result is fetched, and a claimed id is
# remembered so a second get() on the same job is not billed again. Both
# ledgers evict their oldest entries; an evicted unclaimed job is still metered
# on get(), with the application read from its queue URL.
TRACKED_JOB_CAPACITY = 10_000
CLAIMED_ID_CAPACITY = 10_000
KNOWN_APPLICATION_CAPACITY = 10_000

_REQUESTS_SEGMENT = "/requests/"


def application_from_queue_url(response_url: str) -> str:
    head, found, _ = urlparse(response_url).path.rpartition(_REQUESTS_SEGMENT)
    application = head.strip("/")
    return application if found and application else UNKNOWN_APPLICATION


class QueuedJobs:
    def __init__(self, tracked_capacity: int = TRACKED_JOB_CAPACITY,
                 claimed_capacity: int = CLAIMED_ID_CAPACITY,
                 application_capacity: int = KNOWN_APPLICATION_CAPACITY):
        self._lock = threading.Lock()
        self._tracked: "OrderedDict[str, FalCall]" = OrderedDict()
        self._claimed: "OrderedDict[str, None]" = OrderedDict()
        self._applications: "OrderedDict[str, str]" = OrderedDict()
        self._tracked_capacity = tracked_capacity
        self._claimed_capacity = claimed_capacity
        self._application_capacity = application_capacity

    def track(self, request_id: str, call: FalCall) -> None:
        with self._lock:
            self._claimed.pop(request_id, None)
            self._put(request_id, call)

    def remember_application(self, request_id: str, application: str) -> None:
        with self._lock:
            self._applications[request_id] = application
            self._applications.move_to_end(request_id)
            self._evict(self._applications, self._application_capacity)

    def application_of(self, request_id: str) -> Optional[str]:
        with self._lock:
            return self._applications.get(request_id)

    def claim(self, request_id: str, untracked: FalCall) -> Optional[FalCall]:
        """The job's call the first time its result is fetched, and None on every later fetch.

        ``untracked`` stands in for a job this process never saw submitted.
        """
        with self._lock:
            if request_id in self._claimed:
                return None
            self._claimed[request_id] = None
            self._evict(self._claimed, self._claimed_capacity)
            self._applications.pop(request_id, None)
            call = self._tracked.pop(request_id, None)
        return call if call is not None else untracked

    def _put(self, request_id: str, call: FalCall) -> None:
        self._tracked[request_id] = call
        self._tracked.move_to_end(request_id)
        self._evict(self._tracked, self._tracked_capacity)

    @staticmethod
    def _evict(ledger: OrderedDict, capacity: int) -> None:
        while len(ledger) > capacity:
            ledger.popitem(last=False)


queued_jobs = QueuedJobs()
