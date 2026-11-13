"""``no core override`` judges a name by the scope that binds it and a method call by the object it
is called on: a function's local ``key = <core value>`` does not make every ``key`` in the module
core, and a plugin method that ``setattr``s its parameter does not make ``set.add(x)`` a patcher.
The real rebinds those shapes sit next to stay caught."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.plugin_validate_core_override import core_override_findings


def _plugin(root: Path, init_py: str) -> Path:
    root.mkdir()
    (root / "__init__.py").write_text(init_py, encoding="utf-8")
    return root


def test_a_function_local_core_value_does_not_taint_the_module(tmp_path):
    plugin = _plugin(tmp_path / "p", (
        "def _host_key():\n"
        "    from agent.context_compressor import MODEL_ONLY_KEY\n"
        "    return MODEL_ONLY_KEY\n"
        "def stamp(rows):\n"
        "    key = _host_key()\n"
        "    return [r.get(key) for r in rows]\n"
        "class Scheduler:\n"
        "    def schedule(self, key):\n"
        "        self.key = key\n"
        "_SCHED = Scheduler()\n"
        "def run(key):\n"
        "    _SCHED.schedule(key)\n"
        "    key.x = 1\n"
        "def init():\n"
        "    global _AGENT\n"
        "    import importlib\n"
        "    _AGENT = importlib.import_module('run_agent')\n"
        "def patch():\n"
        "    _AGENT.AIAgent.run_conversation = print\n"
    ))

    assert core_override_findings(plugin) == [
        "_AGENT.AIAgent.run_conversation (__init__.py:19)",
    ]


def test_a_same_named_method_on_an_unrelated_object_is_not_a_patcher(tmp_path):
    plugin = _plugin(tmp_path / "p", (
        "import run_agent\n"
        "class Recorder:\n"
        "    def add(self, target, fn):\n"
        "        setattr(target, 'hook', fn)\n"
        "class Sub(Recorder):\n"
        "    pass\n"
        "def register(ctx):\n"
        "    seen = set()\n"
        "    seen.add(run_agent)\n"
        "    {run_agent}.add(run_agent)\n"
        "    rec = Sub()\n"
        "    rec.add(run_agent, print)\n"
        "    ctx.add(run_agent, print)\n"
    ))

    assert core_override_findings(plugin) == [
        "rec.add(run_agent, ...) (__init__.py:12)",
        "ctx.add(run_agent, ...) (__init__.py:13)",
    ]
