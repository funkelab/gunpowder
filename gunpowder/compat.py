import sys

PY2 = sys.version_info[0] == 2

if PY2:
    binary_type = str
else:
    binary_type = bytes

def ensure_str(s):
    if PY2:
        if isinstance(s, buffer):
            s = str(s)
    else:
        if isinstance(s, memoryview):
            s = s.tobytes()
        if isinstance(s, binary_type):
            s = s.decode('ascii')
    return s
