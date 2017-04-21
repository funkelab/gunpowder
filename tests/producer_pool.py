from gunpowder import ProducerPool
import time
import random
import os

def long_lasting_task():
    time.sleep(random.randint(1,5))
    # if random.random() > 0.8:
        # raise RuntimeError("something bad happened in process " + str(os.getpid()))
    time.sleep(random.randint(1,5))
    return os.getpid()

def run():

    print("Main process started with PID " + str(os.getpid()))

    pool = ProducerPool([long_lasting_task]*2)
    print("Starting work force...")
    pool.start()
    print("Work force started...")

    print("Getting results. Try eviling around to see what happens with the work force processes.")
    for i in range(100):
        result = pool.get()
        print("got: " + str(result))

    print("Stopping work force...")
    pool.stop()

if __name__ == "__main__":
    run()
