#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run with Gunicorn using our configuration
gunicorn -c gunicorn_config.py src.app:app
