#!/usr/bin/env python3
"""
Test runner for Portfolio Dashboard
"""

import sys
import os
import subprocess
import argparse

def run_tests(test_path=None, verbose=False, coverage=False):
    """Run the test suite"""
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "portfolio_management"))
    sys.path.insert(0, os.path.join(project_root, "shared"))
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=portfolio_management", "--cov-report=html", "--cov-report=term"])
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    # Add additional pytest options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Portfolio Dashboard tests")
    parser.add_argument("--test", "-t", help="Specific test file or directory to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    
    args = parser.parse_args()
    
    print("Portfolio Dashboard Test Suite")
    print("=" * 60)
    
    exit_code = run_tests(
        test_path=args.test,
        verbose=args.verbose,
        coverage=args.coverage
    )
    
    if exit_code == 0:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[ERROR] Some tests failed!")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
