import threading

BATCH_STEP = threading.Event()

def step_next(event):
    BATCH_STEP.set()

def wait_for_step():
    BATCH_STEP.wait()
    BATCH_STEP.clear()