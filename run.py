#!/usr/bin/env python3
"""
Application Launcher for Wild Market Capital Portfolio Manager.

This script provides an easy way to launch the Streamlit application
with proper configuration and environment setup.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        logger.error(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        logger.info(f"Python version: {sys.version.split()[0]} ✅")


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'yfinance'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            logger.debug(f"{package}: ✅")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"{package}: ❌")

    if missing_packages:
        logger.error("Missing required packages:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        logger.error("\nInstall missing packages with:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    else:
        logger.info("All dependencies available ✅")


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        PROJECT_ROOT / 'data' / 'portfolios',
        PROJECT_ROOT / 'exports',
        PROJECT_ROOT / 'logs',
        PROJECT_ROOT / 'cache'
    ]

    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")

            # Create .gitkeep file
            gitkeep = directory / '.gitkeep'
            if not gitkeep.exists():
                gitkeep.touch()

    logger.info("Directory structure ready ✅")


def setup_environment():
    """Setup environment variables and configuration."""
    # Add project root to Python path
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Set environment variables for the app
    os.environ['PORTFOLIO_MANAGER_ROOT'] = project_root_str
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    logger.info("Environment configured ✅")


def get_streamlit_config():
    """Get Streamlit configuration arguments."""
    config = [
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--theme.base', 'dark',
        '--theme.primaryColor', '#BF9FFB',
        '--theme.backgroundColor', '#0D1015',
        '--theme.secondaryBackgroundColor', '#2A2E39',
        '--theme.textColor', '#FFFFFF'
    ]
    return config


def launch_streamlit(dev_mode=False, port=8501, host='localhost'):
    """Launch the Streamlit application."""
    app_path = PROJECT_ROOT / 'streamlit_app' / 'app.py'

    if not app_path.exists():
        logger.error(f"Application file not found: {app_path}")
        sys.exit(1)

    # Build streamlit command
    cmd = ['streamlit', 'run', str(app_path)]

    # Add configuration
    config = get_streamlit_config()
    cmd.extend(config)

    # Override port and host if specified
    if port != 8501:
        # Remove default port config and add custom
        cmd = [arg for arg in cmd if not (arg in ['--server.port', '8501'])]
        cmd.extend(['--server.port', str(port)])

    if host != 'localhost':
        cmd = [arg for arg in cmd if not (arg in ['--server.address', '0.0.0.0'])]
        cmd.extend(['--server.address', host])

    if dev_mode:
        logger.info("🚀 Starting in DEVELOPMENT mode")
        cmd.extend(['--server.runOnSave', 'true'])
    else:
        logger.info("🚀 Starting in PRODUCTION mode")

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"URL: http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server")

    try:
        # Launch Streamlit
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    logger.info("🧪 Running test suite...")

    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests/', '-v'],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode == 0:
            logger.info("✅ All tests passed!")
        else:
            logger.error("❌ Some tests failed")
            sys.exit(1)

    except FileNotFoundError:
        logger.error("pytest not found. Install with: pip install pytest")
        sys.exit(1)


def show_system_info():
    """Display system information and status."""
    print("🚀 Wild Market Capital Portfolio Manager")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Python Version: {sys.version.split()[0]}")

    # Check if virtual environment is active
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual Environment: ✅ Active")
    else:
        print("Virtual Environment: ⚠️  Not detected")

    print(f"Current Directory: {os.getcwd()}")

    # Check key files
    key_files = [
        'requirements.txt',
        'streamlit_app/app.py',
        'core/data_manager/__init__.py'
    ]

    print("\nKey Files:")
    for file_path in key_files:
        full_path = PROJECT_ROOT / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"  {file_path}: {status}")

    # Check directories
    print("\nDirectories:")
    directories = ['data/portfolios', 'exports', 'tests', 'core']
    for dir_path in directories:
        full_path = PROJECT_ROOT / dir_path
        status = "✅" if full_path.exists() else "❌"
        print(f"  {dir_path}: {status}")

    print("\n" + "=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Wild Market Capital Portfolio Manager Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Launch application
  python run.py --dev             # Launch in development mode
  python run.py --port 8080       # Launch on custom port
  python run.py --test            # Run test suite
  python run.py --info            # Show system information

For more information, visit: https://github.com/wildmarketcapital/portfolio-manager
        """
    )

    parser.add_argument(
        '--dev',
        action='store_true',
        help='Run in development mode with auto-reload'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run the server on (default: 8501)'
    )

    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to bind the server to (default: localhost)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run the test suite instead of launching the app'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='Check system requirements and exit'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle info command
    if args.info:
        show_system_info()
        return

    # System checks
    logger.info("Performing system checks...")
    check_python_version()

    if args.check:
        check_dependencies()
        setup_directories()
        logger.info("✅ System check complete!")
        return

    # Handle test command
    if args.test:
        check_dependencies()
        run_tests()
        return

    # Normal application launch
    logger.info("🚀 Launching Wild Market Capital Portfolio Manager...")

    try:
        check_dependencies()
        setup_directories()
        setup_environment()

        # Launch application
        launch_streamlit(
            dev_mode=args.dev,
            port=args.port,
            host=args.host
        )

    except KeyboardInterrupt:
        logger.info("\n👋 Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        logger.error("Try running with --check to diagnose issues")
        sys.exit(1)


if __name__ == '__main__':
    main()