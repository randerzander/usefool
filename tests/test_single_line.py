import sys
import time
import threading

def test_method(name, update_fn, duration=3):
    print(f"\nTesting Method: {name}")
    print("-" * 30)
    start = time.time()
    while time.time() - start < duration:
        elapsed = time.time() - start
        update_fn(elapsed)
        time.sleep(0.1)
    # Clear line at end
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()
    print(f"✓ {name} completed.")

def method_carriage_return(elapsed):
    # Method 1: Basic carriage return
    sys.stdout.write(f"\r[Method 1] Elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

def method_ansi_clear(elapsed):
    # Method 2: ANSI Clear Line + carriage return
    # \x1b[2K clears the entire current line
    sys.stdout.write(f"\x1b[2K\r[Method 2] Elapsed: {elapsed:.1f}s")
    sys.stdout.flush()

def method_space_padding(elapsed):
    # Method 3: Carriage return + padding spaces to overwrite old text
    msg = f"[Method 3] Elapsed: {elapsed:.1f}s"
    sys.stdout.write(f"\r{msg}{" " * (60 - len(msg))}")
    sys.stdout.flush()

if __name__ == "__main__":
    print("Terminal Single-Line Update Test")
    print("===============================")
    
    try:
        test_method("Simple \\r", method_carriage_return)
        test_method("ANSI Clear (\\x1b[2K\\r)", method_ansi_clear)
        test_method("Space Padding", method_space_padding)
        
        print("\nAll tests finished. If any of these produced multiple lines, please let me know which ones!")
    except KeyboardInterrupt:
        print("\nTest cancelled.")
