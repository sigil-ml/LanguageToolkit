test-filter:
	pytest tests/test_string_filter.py --capture=no

functional-test:
	pytest src/language_toolkit/tests/string_filter_tests/test_string_filter_functional.py --capture=no