import argparse
import unittest
import sys
import os

# Adjust path to allow importing from current directory when run as a script
sys.path.append(os.path.dirname(__file__))

# Import our live fire test function
try:
    from test_live_fire import run_live_fire_test
except ImportError:
    from .test_live_fire import run_live_fire_test

try:
    from test_session_recovery import run_session_recovery_test
except ImportError:
    from .test_session_recovery import run_session_recovery_test

def run_unit_tests():
    print("\n" + "="*20 + " RUNNING UNIT TESTS " + "="*20)
    test_dir = os.path.dirname(__file__)
    loader = unittest.TestLoader()
    # Discover tests in the same directory as this script
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description="Arif Main Test Suite")
    parser.add_argument("--unit", action="store_true", help="Run unit tests (no network calls)")
    parser.add_argument("--live", action="store_true", help="Run live-fire connectivity and guard tests")
    parser.add_argument("--cli", nargs="+", help="Specific CLIs to test for live-fire (default: all)")
    parser.add_argument("--all", action="store_true", help="Run all tests (unit + live-fire)")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    success = True

    if args.unit or args.all:
        if not run_unit_tests():
            success = False

    if args.live or args.all:
        print("\n" + "="*20 + " RUNNING LIVE-FIRE TESTS " + "="*20)
        # try:
        #     run_live_fire_test(clis=args.cli)
        # except Exception as e:
        #     print(f"Live-fire test suite failed: {e}")
        #     success = False
        try:
            run_live_fire_test(clis=args.cli)
        except Exception as e:
            print(f"Live-fire test suite failed: {e}")
            success = False

        try:
            run_session_recovery_test(clis=args.cli)
        except Exception as e:
            print(f"Session recovery test suite failed: {e}")
            success = False

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
