import sys

# This object detects exit signal and exit all the threads nicely
class SignalHandler:
    stopper = None
    threads = None

    def __init__(self, stopper, threads):
        self.stopper = stopper  # stopper is a threading.Event object
        self.threads = threads  # current running threads

    # This function will be called by python signal module
    # https://docs.python.org/2.7/library/signal.html
    def __call__(self, signum, frame):
        print("called")
        self.stopper.set()
        for thread in self.threads:
            thread.join()
        sys.exit(0)
