#!/bin/bash
set -e

# --- Helper Functions ---
echo_green() {
    echo -e "\033[0;32m$1\033[0m"
}

echo_red() {
    echo -e "\033[0;31m$1\033[0m"
}

# --- Main Script ---

# 1. Check for .env file
if [ ! -f .env ]; then
    echo_red "ERROR: .env file not found."
    echo "Please copy the .env.example file to .env and ensure it points to your local Neo4j and Ollama services."
    echo "  cp .env.example .env"
    exit 1
fi

echo_green "Assuming Neo4j and Ollama are running locally."
echo "Ensure the 'mistral' model is available in your Ollama instance."

# 2. Install Python dependencies
echo_green "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 3. Apply Django database migrations
echo_green "Applying database migrations..."
./manage.py migrate

# 4. Ingest data into the Knowledge Graph
echo_green "Ingesting data into the Knowledge Graph. This is a one-time setup and can be lengthy..."
# Use the --clear flag to ensure a fresh start
./manage.py ingest_data --clear

# 5. Start the Django development server
echo_green "Starting the Django development server at http://0.0.0.0:8000/"
echo "You can now access the application in your browser."
./manage.py runserver 0.0.0.0:8000