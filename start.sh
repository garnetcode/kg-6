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
    echo "Please copy the .env.example file to .env and configure it before running this script."
    echo "  cp .env.example .env"
    exit 1
fi

# 2. Start Docker services
echo_green "Starting Neo4j and Ollama services with Docker Compose..."
sudo docker compose up -d

# Give services a moment to initialize
echo "Waiting for services to start..."
sleep 10

# 3. Pull the Mistral model into Ollama
echo_green "Pulling the 'mistral' model into Ollama (this may take a while)..."
sudo docker exec ollama ollama pull mistral

# 4. Install Python dependencies
echo_green "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# 5. Apply Django database migrations
echo_green "Applying database migrations..."
./manage.py migrate

# 6. Ingest data into the Knowledge Graph
echo_green "Ingesting data into the Knowledge Graph. This is a one-time setup and can be lengthy..."
# Use the --clear flag to ensure a fresh start
./manage.py ingest_data --clear

# 7. Start the Django development server
echo_green "Starting the Django development server at http://0.0.0.0:8000/"
echo "You can now access the application in your browser."
./manage.py runserver 0.0.0.0:8000