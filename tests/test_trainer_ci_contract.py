from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_fetches_complete_audited_engine_main_history():
    workflow = (ROOT / ".github" / "workflows" / "trainer-tests.yml").read_text(
        encoding="utf-8"
    )
    command = (
        "git -C external/Atomic-Stockfish fetch --no-tags --depth=2147483647 "
        "origin +refs/heads/main:refs/remotes/origin/main"
    )

    # Both CPU and trusted CUDA jobs must make the ancestry proof meaningful.
    assert workflow.count(command) == 2
