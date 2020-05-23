PYTHON ?= python3

.PHONY: test bench plot clean

test:
	$(PYTHON) -m pytest -q

bench:
	$(PYTHON) scripts/run_benchmarks.py --out results/results.csv

plot:
	$(PYTHON) scripts/plot_results.py results/results.csv --out-dir results

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
