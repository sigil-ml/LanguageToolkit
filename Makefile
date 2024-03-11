simple-test:
	pytest src/language_toolkit/tests/simple_test_model.py --capture=no

functional-test:
	pytest src/language_toolkit/tests/string_filter_tests/test_string_filter_functional.py --capture=no


