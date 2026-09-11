"""
Tests for dlab.docker module.

These tests require Docker to be running.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

from dlab.docker import (
    _build_interactive_run_cmd,
    _run_docker_build,
    build_image,
    build_runner_script,
    compute_docker_dir_hash,
    container_exists,
    exec_command,
    image_exists,
    run_interactive_container,
    start_container,
    stop_container,
)


TEST_IMAGE_NAME: str = "dlab-test-image"
TEST_CONTAINER_NAME: str = "dlab-test-container"


@pytest.fixture
def docker_dir(tmp_path: Path) -> Path:
    """Create a minimal docker directory with a Dockerfile."""
    docker_path: Path = tmp_path / "docker"
    docker_path.mkdir()

    dockerfile: Path = docker_path / "Dockerfile"
    dockerfile.write_text('FROM python:3.11-slim\nCMD ["echo", "hello"]\n')

    return tmp_path


@pytest.fixture
def built_image(docker_dir: Path) -> str:
    """Build a test image and return its name. Cleans up after test."""
    build_image(str(docker_dir), TEST_IMAGE_NAME)
    yield TEST_IMAGE_NAME
    # Cleanup: remove the image after test
    import subprocess

    subprocess.run(["docker", "rmi", "-f", TEST_IMAGE_NAME], capture_output=True)


@pytest.fixture
def running_container(built_image: str, tmp_path: Path) -> str:
    """Start a test container and return its name. Cleans up after test."""
    work_dir: Path = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "_opencode_logs").mkdir()

    start_container(built_image, str(work_dir), TEST_CONTAINER_NAME)
    yield TEST_CONTAINER_NAME
    # Cleanup: stop and remove container after test
    stop_container(TEST_CONTAINER_NAME)


class TestImageExists:
    """Tests for image_exists function."""

    def test_nonexistent_image(self) -> None:
        """Should return False for image that doesn't exist."""
        result: bool = image_exists("this-image-definitely-does-not-exist-abc123")
        assert result is False

    def test_existing_image(self, built_image: str) -> None:
        """Should return True for image that exists."""
        result: bool = image_exists(built_image)
        assert result is True


class TestBuildImage:
    """Tests for build_image function."""

    def test_build_success(self, docker_dir: Path) -> None:
        """Should build image successfully."""
        image_name: str = "dlab-build-test"

        build_image(str(docker_dir), image_name)

        assert image_exists(image_name)

        # Cleanup
        import subprocess

        subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)

    def test_missing_docker_dir(self, tmp_path: Path) -> None:
        """Should raise ValueError if docker/ directory missing."""
        with pytest.raises(ValueError, match="docker/ directory not found"):
            build_image(str(tmp_path), "test-image")

    def test_invalid_dockerfile(self, tmp_path: Path) -> None:
        """Should raise ValueError on build failure."""
        docker_path: Path = tmp_path / "docker"
        docker_path.mkdir()
        (docker_path / "Dockerfile").write_text("INVALID DOCKERFILE CONTENT\n")

        with pytest.raises(ValueError, match="Docker build failed"):
            build_image(str(tmp_path), "test-image")

    def test_build_error_includes_output(self, tmp_path: Path) -> None:
        """Build failures should include diagnostic output, not an empty string."""
        docker_path: Path = tmp_path / "docker"
        docker_path.mkdir()
        (docker_path / "Dockerfile").write_text("FROM nonexistent-image-xxxxx\n")

        returncode, output = _run_docker_build(
            ["docker", "build", str(docker_path)],
        )
        assert returncode != 0
        assert len(output) > 0, "Build error output should not be empty"


class TestContainerExists:
    """Tests for container_exists function."""

    def test_nonexistent_container(self) -> None:
        """Should return False for container that doesn't exist."""
        result: bool = container_exists("this-container-definitely-does-not-exist")
        assert result is False

    def test_existing_container(self, running_container: str) -> None:
        """Should return True for container that exists."""
        result: bool = container_exists(running_container)
        assert result is True


class TestStartContainer:
    """Tests for start_container function."""

    def test_start_success(self, built_image: str, tmp_path: Path) -> None:
        """Should start container with correct mounts."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()
        container_name: str = "dlab-start-test"

        try:
            start_container(built_image, str(work_dir), container_name)
            assert container_exists(container_name)
        finally:
            stop_container(container_name)

    def test_container_already_exists(
        self, running_container: str, built_image: str, tmp_path: Path
    ) -> None:
        """Should raise ValueError if container already exists."""
        work_dir: Path = tmp_path / "work2"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()

        with pytest.raises(ValueError, match="Container already exists"):
            start_container(built_image, str(work_dir), running_container)


class TestExecCommand:
    """Tests for exec_command function."""

    def test_exec_success(self, running_container: str) -> None:
        """Should execute command and return result."""
        exit_code, stdout, stderr = exec_command(
            running_container, ["echo", "hello", "world"]
        )

        assert exit_code == 0
        assert "hello world" in stdout

    def test_exec_failure(self, running_container: str) -> None:
        """Should return non-zero exit code on failure."""
        exit_code, stdout, stderr = exec_command(
            running_container,
            ["false"],  # 'false' command always exits with 1
        )

        assert exit_code == 1


class TestStopContainer:
    """Tests for stop_container function."""

    def test_stop_running_container(self, built_image: str, tmp_path: Path) -> None:
        """Should stop and remove a running container."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()
        container_name: str = "dlab-stop-test"

        start_container(built_image, str(work_dir), container_name)
        assert container_exists(container_name)

        stop_container(container_name)
        assert not container_exists(container_name)

    def test_stop_nonexistent_container(self) -> None:
        """Should not raise error if container doesn't exist (idempotent)."""
        # Should not raise
        stop_container("this-container-does-not-exist")


class TestStartContainerWithEnvFile:
    """Tests for start_container with env_file parameter."""

    def test_start_with_env_file(self, built_image: str, tmp_path: Path) -> None:
        """Should start container with env file and have env vars available."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()
        container_name: str = "dlab-env-test"

        env_file: Path = tmp_path / "test.env"
        env_file.write_text("TEST_VAR=hello_from_env\n")

        try:
            start_container(
                built_image, str(work_dir), container_name, env_file=str(env_file)
            )
            assert container_exists(container_name)

            # Verify env var is available in container
            exit_code, stdout, stderr = exec_command(
                container_name, ["sh", "-c", "echo $TEST_VAR"]
            )
            assert exit_code == 0
            assert "hello_from_env" in stdout
        finally:
            stop_container(container_name)

    def test_start_with_nonexistent_env_file(
        self, built_image: str, tmp_path: Path
    ) -> None:
        """Should raise ValueError if env file doesn't exist."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()
        container_name: str = "dlab-env-missing-test"

        with pytest.raises(ValueError, match="Environment file not found"):
            start_container(
                built_image,
                str(work_dir),
                container_name,
                env_file="/path/to/nonexistent/env/file",
            )

    def test_start_without_env_file(self, built_image: str, tmp_path: Path) -> None:
        """Should start container without env file (default behavior)."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()
        container_name: str = "dlab-no-env-test"

        try:
            start_container(built_image, str(work_dir), container_name, env_file=None)
            assert container_exists(container_name)
        finally:
            stop_container(container_name)


class TestStartContainerWithOpenCodeData:
    """Tests for mounting dedicated OpenCode authentication data."""

    def test_start_with_opencode_data_dir(
        self, built_image: str, tmp_path: Path
    ) -> None:
        """Should mount the host OpenCode data directory in the container."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()
        opencode_data_dir: Path = tmp_path / ".docker-opencode-data"
        opencode_data_dir.mkdir()
        (opencode_data_dir / "auth.json").write_text("oauth-data")
        container_name: str = "dlab-opencode-data-test"

        try:
            start_container(
                built_image,
                str(work_dir),
                container_name,
                opencode_data_dir=str(opencode_data_dir),
            )
            exit_code, stdout, stderr = exec_command(
                container_name,
                ["cat", "/root/.local/share/opencode/auth.json"],
            )
            assert exit_code == 0
            assert stdout == "oauth-data"
        finally:
            stop_container(container_name)

    def test_start_with_missing_opencode_data_dir(
        self, built_image: str, tmp_path: Path
    ) -> None:
        """Should reject a missing OpenCode data directory."""
        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()

        with pytest.raises(ValueError, match="OpenCode data directory not found"):
            start_container(
                built_image,
                str(work_dir),
                "dlab-opencode-data-missing-test",
                opencode_data_dir=str(tmp_path / "missing"),
            )


class TestOAuthEnvOverrideIntegration:
    """
    Real-Docker proof that get_oauth_env_overrides() actually blanks a
    conflicting .env token inside a running container (not just in-memory).
    """

    def test_oauth_override_blanks_github_token_in_container(
        self, built_image: str, tmp_path: Path
    ) -> None:
        from dlab.model_fallback import get_oauth_env_overrides

        work_dir: Path = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "_opencode_logs").mkdir()

        env_file: Path = tmp_path / "test.env"
        env_file.write_text("GITHUB_TOKEN=stale-invalid-token\n")

        opencode_data_dir: Path = tmp_path / "oauth-data"
        opencode_data_dir.mkdir()
        (opencode_data_dir / "auth.json").write_text(
            '{"github-copilot-enterprise": {"type": "oauth"}}'
        )

        container_name: str = "dlab-oauth-override-test"
        extra_env: dict[str, str] = get_oauth_env_overrides(str(opencode_data_dir))
        assert extra_env == {"GITHUB_TOKEN": ""}

        try:
            start_container(
                built_image,
                str(work_dir),
                container_name,
                env_file=str(env_file),
                extra_env=extra_env,
                opencode_data_dir=str(opencode_data_dir),
            )
            exit_code, stdout, stderr = exec_command(
                container_name,
                ["sh", "-c", "echo $GITHUB_TOKEN"],
            )
            assert exit_code == 0
            assert stdout.strip() == ""
        finally:
            stop_container(container_name)


class TestRunInteractiveContainer:
    """Tests for run_interactive_container() and its command construction."""

    def test_build_cmd_has_tty_and_rm_flags(self) -> None:
        cmd: list[str] = _build_interactive_run_cmd(
            "some-image", ["opencode", "auth", "login"]
        )
        assert cmd[:4] == ["docker", "run", "--rm", "-it"]
        assert cmd[-3:] == ["opencode", "auth", "login"]
        assert "some-image" in cmd

    def test_build_cmd_includes_env_file_and_mount(self, tmp_path: Path) -> None:
        env_file: Path = tmp_path / "test.env"
        env_file.write_text("KEY=value\n")
        opencode_data_dir: Path = tmp_path / "oauth-data"
        opencode_data_dir.mkdir()

        cmd: list[str] = _build_interactive_run_cmd(
            "some-image",
            ["true"],
            env_file=str(env_file),
            opencode_data_dir=str(opencode_data_dir),
        )
        assert "--env-file" in cmd
        assert str(env_file.resolve()) in cmd
        assert f"{opencode_data_dir.resolve()}:/root/.local/share/opencode" in cmd

    def test_build_cmd_rejects_missing_opencode_data_dir(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="OpenCode data directory not found"):
            _build_interactive_run_cmd(
                "some-image",
                ["true"],
                opencode_data_dir=str(tmp_path / "missing"),
            )

    def test_build_cmd_rejects_missing_env_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Environment file not found"):
            _build_interactive_run_cmd(
                "some-image",
                ["true"],
                env_file=str(tmp_path / "missing.env"),
            )

    def test_runs_and_returns_exit_code(self, built_image: str) -> None:
        """
        A trivial command should run and exit cleanly.

        Skipped without a real terminal: `docker run -it` requires stdin to
        be an actual TTY (verified: it errors "the input device is not a
        TTY" under pytest's non-interactive stdin), so this only runs when
        invoked from a real terminal session.
        """
        if not sys.stdin.isatty():
            pytest.skip("docker run -it requires a real terminal (stdin is not a TTY)")
        exit_code: int = run_interactive_container(built_image, ["true"])
        assert exit_code == 0


class TestBuildRunnerScript:
    """Tests for build_runner_script function (no Docker required)."""

    def test_basic_script(self) -> None:
        """Script should have opencode command and tee."""
        script: str = build_runner_script(
            "/.prompt.txt", "anthropic/claude-sonnet-4-0", "main"
        )

        assert "#!/bin/bash" in script
        assert "set -o pipefail" in script
        assert (
            'opencode run --format json --log-level DEBUG --model "anthropic/claude-sonnet-4-0"'
            in script
        )
        # dlab_start JSON line is written first, then opencode appends via tee -a
        assert "'type':'dlab_start'" in script
        assert "tee -a /_opencode_logs/main.log" in script

    def test_script_model_and_prefix(self) -> None:
        """Model and log prefix should appear correctly in script."""
        script: str = build_runner_script(
            "/.prompt.txt",
            "anthropic/claude-opus-4-0",
            "instance-3",
        )

        assert '--model "anthropic/claude-opus-4-0"' in script
        assert "tee -a /_opencode_logs/instance-3.log" in script


class TestComputeDockerDirHash:
    """Tests for compute_docker_dir_hash function (no Docker required)."""

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same directory should produce same hash on repeated calls."""
        docker_dir: Path = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM python:3.11-slim\n")

        hash1: str = compute_docker_dir_hash(docker_dir)
        hash2: str = compute_docker_dir_hash(docker_dir)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest

    def test_detects_file_change(self, tmp_path: Path) -> None:
        """Modifying a file should change the hash."""
        docker_dir: Path = tmp_path / "docker"
        docker_dir.mkdir()
        dockerfile: Path = docker_dir / "Dockerfile"
        dockerfile.write_text("FROM python:3.11-slim\n")

        hash1: str = compute_docker_dir_hash(docker_dir)

        dockerfile.write_text("FROM python:3.12-slim\n")
        hash2: str = compute_docker_dir_hash(docker_dir)

        assert hash1 != hash2

    def test_detects_new_file(self, tmp_path: Path) -> None:
        """Adding a file should change the hash."""
        docker_dir: Path = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM python:3.11-slim\n")

        hash1: str = compute_docker_dir_hash(docker_dir)

        (docker_dir / "requirements.txt").write_text("pandas\n")
        hash2: str = compute_docker_dir_hash(docker_dir)

        assert hash1 != hash2

    def test_ignores_pycache(self, tmp_path: Path) -> None:
        """__pycache__ and .pyc files should be ignored."""
        docker_dir: Path = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM python:3.11-slim\n")

        hash1: str = compute_docker_dir_hash(docker_dir)

        # Add __pycache__ directory with .pyc file
        pycache: Path = docker_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "module.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

        # Add a standalone .pyc file
        (docker_dir / "other.pyc").write_bytes(b"\x00\x01\x02")

        hash2: str = compute_docker_dir_hash(docker_dir)

        assert hash1 == hash2

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return a valid hash without crashing."""
        docker_dir: Path = tmp_path / "docker"
        docker_dir.mkdir()

        result: str = compute_docker_dir_hash(docker_dir)

        assert isinstance(result, str)
        assert len(result) == 64

    def test_opencode_version_changes_hash(self, tmp_path: Path) -> None:
        """Different opencode_version should produce a different hash."""
        docker_dir: Path = tmp_path / "docker"
        docker_dir.mkdir()
        (docker_dir / "Dockerfile").write_text("FROM ubuntu:22.04\n")

        hash_latest: str = compute_docker_dir_hash(docker_dir)
        hash_pinned: str = compute_docker_dir_hash(docker_dir, "0.1.50")

        assert hash_latest != hash_pinned
