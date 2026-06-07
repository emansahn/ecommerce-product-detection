# Contributing to Ecommerce Product Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inspiring community for all.

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (venv, conda, etc.)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/emansahn/ecommerce-product-detection.git
cd ecommerce-product-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clean, readable code
- Follow PEP 8 style guide
- Add docstrings to functions and classes
- Add type hints where possible

### 3. Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v
```

### 4. Code Quality

```bash
# Format code with Black
black src/ tests/

# Check with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add new feature" -m "Detailed description of changes"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Commit Message Format

We follow conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat(detection): add batch detection support

Implement batch processing for multiple images to improve inference throughput.
Adds new detect_batch method to ProductDetector class.

Closes #123
```

## Pull Request Guidelines

- Title should be descriptive and follow commit message format
- Include references to related issues (e.g., "Closes #123")
- Add description of changes and motivation
- Include before/after screenshots for UI changes
- Ensure all tests pass
- Request review from maintainers

## Issues

### Reporting Bugs

Use the bug report template and include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error messages/stack traces

### Requesting Features

- Use the feature request template
- Explain the use case
- Describe the desired behavior
- Provide examples if applicable

## Documentation

- Update README for user-facing changes
- Add/update docstrings for new functions
- Update relevant documentation files in `/docs`
- Include code examples for new features

## Review Process

1. Automated tests must pass
2. Code review by maintainers
3. Approval from at least one maintainer
4. Squash and merge to main

## Questions?

- Check existing issues and discussions
- Open a discussion for general questions
- Ask in pull requests for clarification

## Recognition

Contributors will be acknowledged in:
- README contributors section
- Release notes
- GitHub contributors page

Thank you for contributing! 🎉
