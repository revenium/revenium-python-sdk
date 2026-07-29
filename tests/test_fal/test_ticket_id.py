"""ticketId capture for the fal provider (FRONT-1545)."""
import datetime
import os
from unittest.mock import patch

from revenium_middleware.fal.trace_fields import get_ticket_id


class TestTicketIdCapture:
    def test_env_var_fallback(self):
        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'JIRA-77'}):
            assert get_ticket_id({}) == 'JIRA-77'

    def test_metadata_takes_precedence(self):
        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'ENV-1'}):
            assert get_ticket_id({'ticketId': 'META-1'}) == 'META-1'

    def test_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_ticket_id() is None

    def test_build_common_args_excludes_ticket_id(self):
        """ticket_id is completion-only; it must not leak into image/video/audio args."""
        from revenium_middleware.fal._metering import _build_common_args

        with patch.dict(os.environ, {'REVENIUM_TICKET_ID': 'JIRA-77'}):
            args = _build_common_args(
                application="fal-ai/flux/dev",
                request_time_dt=datetime.datetime.now(datetime.timezone.utc),
                usage_metadata={'ticketId': 'META-1'},
                transaction_id="fal-test123",
                is_streamed=False,
            )
            assert 'ticket_id' not in args
