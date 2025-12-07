#!/usr/bin/env python3
"""
Test script for verifying uv virtual environment setup.
This script tests that the "scraper" venv can be created and used correctly.
"""

import os
import subprocess
import sys
import shutil


def run_command(command, shell=True, capture_output=True, use_bash=False):
    """Run a shell command and return the result."""
    if use_bash:
        command = ["bash", "-c", command]
        shell = False
    
    result = subprocess.run(
        command,
        shell=shell,
        capture_output=capture_output,
        text=True
    )
    return result


def test_uv_installed():
    """Test that uv is installed and available."""
    print("\n" + "="*60)
    print("Test 1: Checking if uv is installed...")
    print("="*60)
    
    result = run_command("uv --version")
    if result.returncode != 0:
        print("❌ uv is not installed")
        print("Install uv with: pip install uv")
        return False
    
    version = result.stdout.strip()
    print(f"✓ uv is installed: {version}")
    return True


def test_create_venv():
    """Test creating a virtual environment named 'scraper'."""
    print("\n" + "="*60)
    print("Test 2: Creating uv venv named 'scraper'...")
    print("="*60)
    
    # Clean up any existing venv
    venv_path = "scraper"
    if os.path.exists(venv_path):
        print(f"Removing existing venv at {venv_path}...")
        shutil.rmtree(venv_path)
    
    result = run_command("uv venv scraper")
    if result.returncode != 0:
        print(f"❌ Failed to create venv: {result.stderr}")
        return False
    
    if not os.path.exists(venv_path):
        print("❌ Virtual environment directory was not created")
        return False
    
    print("✓ Virtual environment 'scraper' created successfully")
    
    # Check for activation script
    activate_script = os.path.join(venv_path, "bin", "activate")
    if not os.path.exists(activate_script):
        print(f"❌ Activation script not found at {activate_script}")
        return False
    
    print(f"✓ Activation script found at {activate_script}")
    return True


def test_install_dependencies():
    """Test installing dependencies in the venv."""
    print("\n" + "="*60)
    print("Test 3: Installing dependencies with uv pip...")
    print("="*60)
    
    venv_path = "scraper"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment does not exist")
        return False
    
    # Install dependencies using uv pip
    result = run_command(f"source {venv_path}/bin/activate && uv pip install -r requirements.txt", use_bash=True)
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    
    print("✓ Dependencies installed successfully")
    
    # Verify key packages are installed
    result = run_command(f"source {venv_path}/bin/activate && python -c 'import duckduckgo_search; import pyreadability; import requests'", use_bash=True)
    if result.returncode != 0:
        print(f"❌ Failed to import required packages: {result.stderr}")
        return False
    
    print("✓ All required packages are importable")
    return True


def test_run_tests():
    """Test running the existing test suite in the venv."""
    print("\n" + "="*60)
    print("Test 4: Running test suite in the venv...")
    print("="*60)
    
    venv_path = "scraper"
    if not os.path.exists(venv_path):
        print("❌ Virtual environment does not exist")
        return False
    
    result = run_command(f"source {venv_path}/bin/activate && python test_react_agent.py", use_bash=True)
    if result.returncode != 0:
        print(f"❌ Test suite failed: {result.stderr}")
        return False
    
    print("✓ All tests passed in the venv")
    return True


def cleanup():
    """Clean up the test virtual environment."""
    print("\n" + "="*60)
    print("Cleaning up test environment...")
    print("="*60)
    
    venv_path = "scraper"
    if os.path.exists(venv_path):
        shutil.rmtree(venv_path)
        print(f"✓ Removed test venv at {venv_path}")
    else:
        print("✓ No cleanup needed")


def main():
    """Run all tests."""
    print("="*60)
    print("UV Virtual Environment Test Suite")
    print("="*60)
    print("\nThis script tests the uv venv setup for the scraper project.")
    print("It will create a venv named 'scraper' and verify it works correctly.")
    
    tests = [
        ("uv installation", test_uv_installed),
        ("venv creation", test_create_venv),
        ("dependency installation", test_install_dependencies),
        ("test suite execution", test_run_tests),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n❌ Test '{test_name}' failed")
                # Continue with other tests even if one fails
        except Exception as e:
            failed += 1
            print(f"\n❌ Test '{test_name}' raised an exception: {e}")
    
    # Cleanup
    cleanup()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("="*60)
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
