from lightfall_voice import phrases


def test_read_announcement_names_the_file_stem():
    msg = phrases.tool_announcement("Read", {"file_path": "C:/proj/src/loader.py"})
    assert msg == "Reading loader"


def test_edit_announcement():
    msg = phrases.tool_announcement("Edit", {"file_path": "C:/proj/config.json"})
    assert msg == "Editing config"


def test_bash_announcement_is_generic():
    assert phrases.tool_announcement("Bash", {"command": "rm -rf /"}) == "Running a command"


def test_search_tools():
    assert phrases.tool_announcement("Grep", {}) == "Searching"
    assert phrases.tool_announcement("Glob", {}) == "Searching"


def test_silent_tools_return_none():
    for tool in ("TodoWrite", "Task", "Agent", "ToolSearch"):
        assert phrases.tool_announcement(tool, {}) is None


def test_unknown_tool_is_humanized_not_silent():
    assert phrases.tool_announcement("lightfall_read_device", {}) == "Reading device"
    assert phrases.tool_announcement("lightfall_run_plan", {}) == "Running plan"
    assert phrases.tool_announcement("lightfall_move_motor", {}) == "Moving motor"
    assert phrases.tool_announcement("SomeMcpTool", {}) == "Using some mcp tool"


def test_mcp_prefix_is_stripped():
    assert (
        phrases.tool_announcement("mcp__ipython_tools__lightfall_ipython_execute", {})
        == "Executing ipython"
    )
    assert phrases.tool_announcement("mcp__qt__click_widget", {}) == "Clicking widget"


def test_mcp_prefixed_core_and_silent_tools_use_their_table_entry():
    assert phrases.tool_announcement("mcp__x__Bash", {}) == "Running a command"
    assert phrases.tool_announcement("mcp__lightfall__lightfall_set_emotion", {}) is None


def test_verb_found_mid_name():
    assert (
        phrases.tool_announcement("lightfall_ipython_push_variable", {})
        == "Pushing ipython variable"
    )


def test_permission_phrase_humanizes_mcp_tools():
    assert (
        phrases.permission_phrase("mcp__lightfall__lightfall_move_motor")
        == "Permission needed to use move motor"
    )


def test_permission_phrase_contains_permission_keyword():
    msg = phrases.permission_phrase("Bash")
    assert msg == "Permission needed to run a command"
    assert "permission" in msg.lower()


def test_permission_phrase_with_agent_prefix():
    msg = phrases.permission_phrase("Edit", agent_name="alignment")
    assert msg == "Agent alignment: permission needed to edit a file"


def test_completion_phrase_trims_long_text():
    msg = phrases.completion_phrase("x" * 500)
    assert len(msg) <= 203  # 200 + ellipsis
    assert msg.startswith("x")


def test_completion_phrase_empty_falls_back():
    assert phrases.completion_phrase(None) == "Task complete"
    assert phrases.completion_phrase("   ") == "Task complete"


def test_completion_phrase_strips_markdown_noise():
    assert phrases.completion_phrase("**Done.** See `foo.py`") == "Done. See foo.py"


def test_error_phrase_contains_attention_keyword():
    assert phrases.error_phrase() == "Agent error. Attention needed"
    assert "attention" in phrases.error_phrase().lower()


def test_error_phrase_with_agent():
    assert phrases.error_phrase("alignment") == "Agent alignment error. Attention needed"
