import unittest
import os

def run_all_tests():
    """
    Discover and run all unit tests in the current directory and its subdirectories.
    This function uses the unittest framework to automatically discover all test
    modules that match the pattern '*_tests.py' in the current directory and its
    subdirectories. It then creates a test suite from the discovered tests and runs
    them using a text test runner.
    Returns:
        None
    """
    loader = unittest.defaultTestLoader
    tests = loader.discover(start_dir=".", pattern="*_tests.py")  # Adjust the pattern if necessary

    suite = unittest.TestSuite(tests)

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    os.chdir("./unit_tests")
    run_all_tests()