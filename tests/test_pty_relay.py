import subprocess
import sys
import os
import threading
import pty
import time

def relay_output(master_fd):
    try:
        while True:
            # Read from the master fd of the PTY
            try:
                data = os.read(master_fd, 1024)
                if not data:
                    break
                # Pass directly to stdout buffer
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except OSError:
                break
    except Exception as e:
        print(f"Relay error: {e}")

print("Testing PTY Relay...")
print("If this works, you should see 'Method 1' updating on a SINGLE line below:")
print("-----------------------------------------------------------------------")

# Create a pseudo-terminal
master_fd, slave_fd = pty.openpty()

# Run test_single_line.py as a subprocess using the slave FD
proc = subprocess.Popen(
    [sys.executable, "-u", "test_single_line.py"],
    stdout=slave_fd,
    stderr=slave_fd,
    stdin=slave_fd,
    text=False,
    bufsize=0,
    close_fds=True
)

# Close slave fd in parent
os.close(slave_fd)

# Start relay thread
t = threading.Thread(target=relay_output, args=(master_fd,))
t.start()

proc.wait()
os.close(master_fd)
t.join()

print("\n-----------------------------------------------------------------------")
print("Test Complete.")
