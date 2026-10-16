"""Snapshot contents must survive replay, including Git-ignored binary files."""
import io
import tarfile

import pytest

from benchmarks.checkpoints import canonical_tar, DockerWorkspace


def test_snapshot_preserves_files_modes_and_rejects_unsafe_members(tmp_path):
    source, snapshot = tmp_path / "input.tar", tmp_path / "snapshot.tar"
    with tarfile.open(source, "w") as tar:
        for name, contents in (("./.gitignore", b"*.bin\n"), ("./ignored.bin", b"\x00\xff"),
                               ("./new.sh", b"#!/bin/sh\n"), ("./.git/config", b"private")):
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(contents), 0o755
            tar.addfile(entry, io.BytesIO(contents))
    canonical_tar(source, snapshot)
    with tarfile.open(snapshot) as tar:
        assert set(tar.getnames()) == {".gitignore", "ignored.bin", "new.sh"}
        assert tar.extractfile("ignored.bin").read() == b"\x00\xff"
        assert tar.getmember("new.sh").mode == 0o755
        assert "deleted.py" not in tar.getnames()
    for name, kind in (("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                       ("link", tarfile.SYMTYPE), ("fifo", tarfile.FIFOTYPE)):
        with tarfile.open(source, "w") as tar:
            entry = tarfile.TarInfo(name)
            entry.type = kind
            tar.addfile(entry)
        with pytest.raises(ValueError):
            canonical_tar(source, snapshot)


@pytest.mark.parametrize("root", ["/", "relative", "/app/../root"])
def test_snapshot_rejects_unbounded_container_roots(root):
    with pytest.raises(ValueError, match="bounded absolute"):
        DockerWorkspace("unused", root=root)
