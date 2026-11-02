import socket
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app import find_available_port


def test_find_available_port():
    # Bind a temporary socket to block a port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        blocked_port = s.getsockname()[1]

        # find_available_port should detect that blocked_port is busy and return next free port
        free = find_available_port("127.0.0.1", blocked_port)
        assert free != blocked_port
        assert free > blocked_port
