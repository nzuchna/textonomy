.PHONY: install setup-env-file mlflow-ui quickstart clean run-experiment anon-zip

# Installation
install:
	@echo "📦 Setting up the environment..."
	uv sync --all-extras --locked

# Setup environment file
setup-env-file:
	@if [ ! -f .env ]; then \
		echo "🔑 Creating .env file from template..."; \
		cp .env.example .env; \
		echo "⚠️  Please edit .env and add your OpenAI API key!"; \
	else \
		echo "✅ .env file already exists"; \
	fi

mlflow-ui:
	@echo "🚀 Starting MLflow UI on http://localhost:8080..."
	@if [ ! -d .venv ]; then echo "❌ Virtual environment not found. Run 'make quickstart' first."; exit 1; fi
	@source .venv/bin/activate && mlflow ui --port 8080

# Run experiment
run-experiment:
	@if [ -z "$(config)" ]; then \
		echo "❌ Error: config variable is not set."; \
		echo "Usage: make run-experiment config=<path_to_config_file>"; \
		exit 1; \
	fi
	@echo "🔬 Running experiment with config: $(config)..."
	@source .venv/bin/activate && python scripts/run_experiment.py --config $(config)

# Quick start - Complete setup for new users
quickstart: setup-env-file install
	@echo ""
	@echo "🚀 Textonomy setup complete!"
	@echo ""
	@echo "📋 Next steps:"
	@echo "1. Edit .env and add your keys"
	@echo "2. Activate environment: source .venv/bin/activate"
	@echo "3. Start MLflow UI: make mlflow-ui"
	@echo "4. Open notebooks/lda_reproduction.ipynb"
	@echo ""

# Clean build artifacts
clean:
	rm -rf .pytest_cache __pycache__ .ipynb_checkpoints build dist


# Build an anonymous Code.zip for paper submission.
# Zips ONLY git-tracked files, so nothing identifying or sensitive leaks in
anon-zip:
	@echo "📦 Building anonymous Code.zip from git-tracked files..."
	@rm -f Code.zip
	@git ls-files | zip -X Code.zip -@ > /dev/null
	@echo "✅ Code.zip created ($$(unzip -l Code.zip | tail -1 | awk '{print $$2}') files)."