import unittest
import os

def run_all_tests():
    loader = unittest.defaultTestLoader
    tests = loader.discover(start_dir=".", pattern="*_tests.py")  # Adjust the pattern if necessary

    suite = unittest.TestSuite(tests)

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == "__main__":
    os.chdir("./unit_tests")
    run_all_tests()