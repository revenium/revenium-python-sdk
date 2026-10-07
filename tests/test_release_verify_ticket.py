"""scripts/release_verify_ticket.py: the matrix table, the ticket it files, and its Linear calls (stubbed)."""
import importlib.util
import io
import json
import pathlib

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "release_verify_ticket.py"
if not SCRIPT.exists():
    pytest.skip("scripts/ is internal-only; the public export copies tests/ without it", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("release_verify_ticket", SCRIPT)
ticket = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ticket)

STUB_KEY = "stub-linear-key-for-tests"

JUNIT_WITH_A_RED_CELL = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.entry_points.test_entry_point_matrix"
            name="test_entry_point_emits_one_payload[openai.chat.create.sync]"/>
  <testcase classname="tests.entry_points.test_entry_point_matrix"
            name="test_entry_point_emits_one_payload[litellm.acompletion.async]">
    <failure message="OracleMismatch: expected exactly one payload, got 0 | []">trace</failure>
  </testcase>
  <testcase classname="tests.entry_points.test_entry_point_matrix"
            name="test_entry_point_emits_one_payload[ollama.client.chat.sync]">
    <skipped type="pytest.xfail" message="metered since 0.9.2; 0.9.1 predates it"/>
  </testcase>
  <testcase classname="tests.entry_points.test_resource_inventory"
            name="test_every_public_method_is_registered_or_allowed[openai]">
    <error message="collection error"/>
  </testcase>
</testsuite></testsuites>
"""

JUNIT_ALL_GREEN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
  <testcase classname="tests.entry_points.test_entry_point_matrix"
            name="test_entry_point_emits_one_payload[vertex.chat_session.send_message]"/>
</testsuite></testsuites>
"""


@pytest.fixture
def red_junit(tmp_path):
    path = tmp_path / "junit-default.xml"
    path.write_text(JUNIT_WITH_A_RED_CELL)
    return str(path)


@pytest.fixture
def green_junit(tmp_path):
    path = tmp_path / "junit-vertex.xml"
    path.write_text(JUNIT_ALL_GREEN)
    return str(path)


class FakeLinear:
    """Answers the four GraphQL operations the script sends, and records every request."""

    def __init__(self, existing_issue=None, label_nodes=(), create_success=True, errors=None):
        self.requests = []
        self.existing_issue = existing_issue
        self.label_nodes = list(label_nodes)
        self.create_success = create_success
        self.errors = errors

    def __call__(self, url, body, headers):
        request = json.loads(body)
        self.requests.append({"url": url, "headers": headers, **request})
        if self.errors:
            return {"errors": self.errors}
        query = request["query"]
        if "issueCreate" in query:
            issue = {"identifier": "BACK-9999", "url": "https://linear.app/revenium/issue/BACK-9999"}
            return {"data": {"issueCreate": {"success": self.create_success, "issue": issue}}}
        if "issueLabels" in query:
            return {"data": {"issueLabels": {"nodes": self.label_nodes}}}
        if "teams" in query:
            return {"data": {"teams": {"nodes": [{"id": "team-back-id"}]}}}
        if "issues(" in query:
            nodes = [self.existing_issue] if self.existing_issue else []
            return {"data": {"issues": {"nodes": nodes}}}
        raise AssertionError(f"unexpected query: {query}")

    def operations(self):
        names = []
        for request in self.requests:
            query = request["query"]
            names.append(next(name for name in ("issueCreate", "issueLabels", "teams", "issues(") if name in query))
        return names

    def created_input(self):
        return next(r["variables"]["input"] for r in self.requests if "issueCreate" in r["query"])


def _file_args(junit, *extra):
    return ["file", "--version", "0.9.2", "--junit", f"default={junit}", "--wheel-sha256", "abc123",
            "--run-url", "https://github.com/revenium/revenium-python-sdk-internal/actions/runs/1", *extra]


def _run(argv, environ=None, post=None):
    out = io.StringIO()
    code = ticket.main(argv, environ=environ or {}, post=post or FakeLinear(), out=out)
    return code, out.getvalue()


class TestCells:
    def test_matrix_cells_are_named_by_row_and_other_checks_by_module(self, red_junit):
        cells = ticket.read_cells("default", red_junit)
        assert [(c.name, c.outcome) for c in cells] == [
            ("openai.chat.create.sync", "passed"),
            ("litellm.acompletion.async", "failed"),
            ("ollama.client.chat.sync", "skipped"),
            ("test_resource_inventory::test_every_public_method_is_registered_or_allowed[openai]", "failed"),
        ]

    def test_an_environment_without_a_report_is_a_red_cell(self, green_junit, tmp_path):
        cells = ticket.collect_cells({"vertex": green_junit, "default": str(tmp_path / "absent.xml")})
        red = ticket.red_cells(cells)
        assert [(c.environment, c.name) for c in red] == [("default", "environment")]
        assert "did not install or import" in red[0].detail

    def test_the_table_escapes_pipes_in_details(self, red_junit):
        table = ticket.markdown_table(ticket.read_cells("default", red_junit))
        assert "got 0 \\| []" in table


class TestTable:
    def test_prints_counts_and_every_cell(self, red_junit, green_junit):
        code, out = _run(["table", "--junit", f"default={red_junit}", "--junit", f"vertex={green_junit}"])
        assert code == 0
        assert out.startswith("**2 red**, 2 passed, 1 skipped or expected to fail")
        assert "`vertex.chat_session.send_message`" in out


class TestFile:
    def test_files_nothing_when_every_cell_passes(self, green_junit):
        fake = FakeLinear()
        code, out = _run(_file_args(green_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert code == 0
        assert "nothing to file" in out
        assert fake.requests == []

    def test_without_a_key_prints_the_ticket_and_calls_nothing(self, red_junit):
        fake = FakeLinear()
        code, out = _run(_file_args(red_junit), environ={}, post=fake)
        assert code == 0
        assert fake.requests == []
        payload = json.loads(out.split("It would be:", 1)[1])
        assert payload["title"] == "[python-sdk] Release verify failed for 0.9.2: 2 cell(s) red"
        assert "`litellm.acompletion.async`" in payload["description"]

    def test_dry_run_calls_nothing_even_with_a_key(self, red_junit):
        fake = FakeLinear()
        code, out = _run(_file_args(red_junit, "--dry-run"), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert fake.requests == []
        assert "Dry run" in out

    def test_files_one_unassigned_back_issue_with_the_red_cells(self, red_junit, tmp_path):
        versions = tmp_path / "versions-default.txt"
        versions.write_text("openai==2.54.0\nanthropic==1.9.0\n")
        fake = FakeLinear()
        code, out = _run(_file_args(red_junit, "--provider-versions", f"default={versions}"),
                         environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)

        assert code == 0
        assert fake.operations() == ["issues(", "teams", "issueLabels", "issueCreate"]
        created = fake.created_input()
        assert created["teamId"] == "team-back-id"
        assert created["title"] == "[python-sdk] Release verify failed for 0.9.2: 2 cell(s) red"
        assert "assigneeId" not in created and "labelIds" not in created
        for expected in ("abc123", "actions/runs/1", "openai==2.54.0", "`litellm.acompletion.async`",
                         "test_every_public_method_is_registered_or_allowed[openai]"):
            assert expected in created["description"]
        assert "ollama.client.chat.sync" not in created["description"]
        assert "Filed BACK-9999" in out

    def test_sends_the_key_only_as_the_authorization_header(self, red_junit):
        fake = FakeLinear()
        _run(_file_args(red_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert all(r["url"] == ticket.LINEAR_GRAPHQL_URL for r in fake.requests)
        assert all(r["headers"]["Authorization"] == STUB_KEY for r in fake.requests)
        assert all(STUB_KEY not in json.dumps(r["variables"]) for r in fake.requests)

    def test_an_open_issue_for_the_version_suppresses_a_second(self, red_junit):
        existing = {"identifier": "BACK-1234", "url": "https://linear.app/x/BACK-1234", "title": "t"}
        fake = FakeLinear(existing_issue=existing)
        code, out = _run(_file_args(red_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert code == 0
        assert fake.operations() == ["issues("]
        assert "Already filed: BACK-1234" in out
        search = fake.requests[0]["variables"]
        assert search == {"team": "BACK", "marker": "Release verify failed for 0.9.2:",
                          "closed": ["completed", "canceled"]}

    def test_uses_an_existing_release_verify_label(self, red_junit):
        labels = [{"id": "other-team-label", "team": {"key": "FRONT"}}, {"id": "back-label", "team": {"key": "BACK"}}]
        fake = FakeLinear(label_nodes=labels)
        _run(_file_args(red_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert fake.created_input()["labelIds"] == ["back-label"]

    def test_a_workspace_label_counts(self, red_junit):
        fake = FakeLinear(label_nodes=[{"id": "workspace-label", "team": None}])
        _run(_file_args(red_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert fake.created_input()["labelIds"] == ["workspace-label"]

    def test_a_graphql_error_fails_loudly(self, red_junit):
        fake = FakeLinear(errors=[{"message": "Authentication required"}])
        with pytest.raises(RuntimeError, match="Authentication required"):
            _run(_file_args(red_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)

    def test_an_unsuccessful_create_fails_loudly(self, red_junit):
        fake = FakeLinear(create_success=False)
        with pytest.raises(RuntimeError, match="success=false"):
            _run(_file_args(red_junit), environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)

    def test_a_failed_environment_without_a_report_files_a_ticket(self, green_junit, tmp_path):
        fake = FakeLinear()
        argv = ["file", "--version", "0.9.2", "--junit", f"vertex={green_junit}",
                "--junit", f"default={tmp_path / 'absent.xml'}"]
        _run(argv, environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert "1 cell(s) red" in fake.created_input()["title"]

    def test_rejects_a_malformed_pair(self, red_junit):
        with pytest.raises(SystemExit):
            _run(["file", "--version", "0.9.2", "--junit", red_junit])


class TestFileUnresolved:
    ARGV = ["file-unresolved", "--run-url", "https://github.com/revenium/revenium-python-sdk-internal/actions/runs/7"]

    def test_files_one_issue_naming_the_run(self):
        fake = FakeLinear()
        code, out = _run(self.ARGV, environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert code == 0
        assert fake.operations() == ["issues(", "teams", "issueLabels", "issueCreate"]
        created = fake.created_input()
        assert created["title"] == "[python-sdk] Release verify could not resolve the version to check"
        assert "actions/runs/7" in created["description"]
        assert "assigneeId" not in created

    def test_an_open_issue_with_the_same_title_suppresses_a_second(self):
        existing = {"identifier": "BACK-4321", "url": "https://linear.app/x/BACK-4321", "title": "t"}
        fake = FakeLinear(existing_issue=existing)
        code, out = _run(self.ARGV, environ={"LINEAR_API_KEY": STUB_KEY}, post=fake)
        assert fake.operations() == ["issues("]
        assert fake.requests[0]["variables"]["marker"] == ticket.UNRESOLVED_TITLE
        assert "Already filed: BACK-4321" in out

    def test_without_a_key_prints_the_issue(self):
        fake = FakeLinear()
        code, out = _run(self.ARGV, environ={}, post=fake)
        assert fake.requests == []
        payload = json.loads(out.split("It would be:", 1)[1])
        assert payload["title"] == ticket.UNRESOLVED_TITLE
