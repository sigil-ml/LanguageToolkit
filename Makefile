test-filter:
	pytest tests/test_string_filter.py --capture=no

functional-test:
	pytest tests/test_string_filter_functional.py --capture=no