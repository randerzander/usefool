import subprocess
import sys
import os
import threading

def relay_output(process):
    while True:
        data = process.stdout.read(1)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        elif process.poll() is not None:
            break

print("Testing Subprocess Relay...")
print("If this works, you should see 'Method 1' updating on a SINGLE line below:")
print("-----------------------------------------------------------------------")

# Run test_single_line.py as a subprocess using the exact same flags as the bot
# text=False, bufsize=0 to ensure binary mode and unbuffered IO
proc = subprocess.Popen(
    [sys.executable, "-u", "test_single_line.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=False,
    bufsize=0
)

# Start relay thread
t = threading.Thread(target=relay_output, args=(proc,))
t.start()
t.join()

print("\n-----------------------------------------------------------------------")
print("Test Complete.")
