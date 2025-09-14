"""
Setup script for Wild Market Capital Portfolio Management System.

This script allows installation of the portfolio management system as a Python package,
making it easier to distribute and install across different environments.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


# Read requirements from requirements.txt
def parse_requirements(filename):
    """Parse requirements from requirements file."""
    requirements = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('-'):
                    # Handle pip options like -e
                    if line.startswith('-e'):
                        continue
                    # Remove version constraints for setup.py
                    requirements.append(line.split('>=')[0].split('==')[0])
        return requirements
    except FileNotFoundError:
        return []


# Core requirements
install_requires = [
    # Core Data Science Libraries
    'pandas>=2.1.0',
    'numpy>=1.24.0',
    'matplotlib>=3.7.0',
    'plotly>=5.15.0',
    'seaborn>=0.12.0',

    # Financial Data
    'yfinance>=0.2.20',
    'alpha-vantage>=2.3.1',

    # Optimization & Mathematics
    'scipy>=1.11.0',
    'scikit-learn>=1.3.0',

    # Web Framework
    'streamlit>=1.28.0',

    # Data Processing
    'openpyxl>=3.1.0',
    'xlsxwriter>=3.1.0',
    'python-dateutil>=2.8.0',
    'pytz>=2023.3',

    # Utilities
    'python-dotenv>=1.0.0',
    'pydantic>=2.0.0',
    'click>=8.1.0',
    'tqdm>=4.65.0',
    'loguru>=0.7.0'
]

# Development requirements
dev_requires = [
    # Testing
    'pytest>=7.4.0',
    'pytest-cov>=4.1.0',
    'pytest-mock>=3.11.0',
    'pytest-xdist>=3.3.0',

    # Code Quality
    'black>=23.7.0',
    'mypy>=1.5.0',
    'pylint>=2.17.0',
    'flake8>=6.0.0',
    'isort>=5.12.0',

    # Documentation
    'sphinx>=7.1.0',
    'sphinx-rtd-theme>=1.3.0',

    # Development Tools
    'pre-commit>=3.3.0',
    'jupyter>=1.0.0',
    'ipython>=8.14.0'
]

# API requirements (for future phases)
api_requires = [
    'fastapi>=0.100.0',
    'uvicorn>=0.23.0',
    'python-jose[cryptography]>=3.3.0',
    'passlib[bcrypt]>=1.7.4',
    'python-multipart>=0.0.6'
]

# Database requirements (for future phases)
db_requires = [
    'sqlalchemy>=2.0.0',
    'alembic>=1.12.0',
    'psycopg2-binary>=2.9.0',
    'redis>=4.6.0'
]

# All extras
all_requires = dev_requires + api_requires + db_requires

setup(
    # Basic package information
    name="wmc-portfolio-manager",
    version="1.0.0",
    description="Professional Portfolio Management and Analytics Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author information
    author="Wild Market Capital Development Team",
    author_email="dev@wildmarketcapital.com",

    # Repository information
    url="https://github.com/wildmarketcapital/portfolio-manager",
    project_urls={
        "Bug Tracker": "https://github.com/wildmarketcapital/portfolio-manager/issues",
        "Documentation": "https://docs.wildmarketcapital.com",
        "Source Code": "https://github.com/wildmarketcapital/portfolio-manager",
        "Discord": "https://discord.gg/wildmarketcapital"
    },

    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*"]),

    # Include additional files
    include_package_data=True,
    package_data={
        "streamlit_app": ["*.py", "components/*.py", "assets/*"],
        "": ["*.md", "*.txt", "*.ini", "*.yaml", "*.json"]
    },

    # Python version requirement
    python_requires=">=3.8",

    # Dependencies
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "api": api_requires,
        "db": db_requires,
        "all": all_requires
    },

    # Entry points for command line scripts
    entry_points={
        "console_scripts": [
            "wmc-portfolio=streamlit_app.app:main",
            "wmc-server=streamlit_app.app:main"
        ]
    },

    # Package classification
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",

        # Intended Audience
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",

        # Topic
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",

        # License
        "License :: OSI Approved :: MIT License",

        # Programming Language
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",

        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",

        # Environment
        "Environment :: Web Environment",
        "Environment :: Console",

        # Framework
        "Framework :: Streamlit"
    ],

    # Keywords for PyPI search
    keywords=[
        "portfolio", "management", "finance", "investment", "analytics",
        "risk", "optimization", "trading", "stocks", "bonds", "etf",
        "financial-analysis", "quantitative-finance", "asset-allocation",
        "modern-portfolio-theory", "streamlit", "dashboard"
    ],

    # Zip safety
    zip_safe=False,

    # Additional metadata
    platforms=["any"],

    # Options for different commands
    options={
        "build": {
            "build_base": "build"
        },
        "sdist": {
            "formats": ["gztar", "zip"]
        }
    }
)


# Additional setup for development environment
def setup_development_environment():
    """Setup development environment after installation."""
    import subprocess
    import sys
    from pathlib import Path

    print("Setting up development environment...")

    # Create necessary directories
    directories = [
        "data/portfolios",
        "exports",
        "logs",
        "cache"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Create .gitkeep files
    for directory in directories:
        gitkeep = Path(directory) / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    # Install pre-commit hooks if in development mode
    try:
        subprocess.run(["pre-commit", "install"], check=True, capture_output=True)
        print("✅ Installed pre-commit hooks")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Could not install pre-commit hooks (optional)")

    print("🚀 Development environment setup complete!")


# Custom commands
class DevelopCommand(setuptools.Command):
    """Custom command to setup development environment."""

    description = "Setup development environment"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        setup_development_environment()


# Add custom command if setuptools is available
try:
    import setuptools

    setup.cmdclass = {"develop": DevelopCommand}
except ImportError:
    pass