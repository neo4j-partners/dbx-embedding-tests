#!/bin/bash
#
# Setup Databricks secrets for Neo4j Embedding Load Test
# Reads credentials from .env file and creates secrets in Databricks
#
# Usage:
#   1. Copy .env.sample to .env and fill in your credentials
#   2. Run: ./setup_secrets.sh [scope-name]
#      Examples:
#        ./setup_secrets.sh                      # Uses default: airline-neo4j-secrets
#        ./setup_secrets.sh airline-neo4j-secrets
#        ./setup_secrets.sh my-custom-scope
#   3. Use secrets in notebooks via dbutils.secrets.get()

set -e

# Scope name from argument or default
SCOPE_NAME="${1:-airline-neo4j-secrets}"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_info "Using secret scope: $SCOPE_NAME"

# Check for .env file
if [[ ! -f "$ENV_FILE" ]]; then
    log_error ".env file not found"
    echo "Copy .env.sample to .env and fill in your Neo4j credentials:"
    echo "  cp .env.sample .env"
    exit 1
fi

# Check for databricks CLI
if ! command -v databricks &> /dev/null; then
    log_error "Databricks CLI not found"
    echo "Install with: pip install databricks-cli"
    echo "Or: brew install databricks"
    echo ""
    echo "Then configure with: databricks auth login"
    exit 1
fi

# Load .env file
log_info "Loading credentials from $ENV_FILE"
set -a
source "$ENV_FILE"
set +a

# Validate required variables
missing=()
[[ -z "$NEO4J_HOST" ]] && missing+=("NEO4J_HOST")
[[ -z "$NEO4J_USER" ]] && missing+=("NEO4J_USER")
[[ -z "$NEO4J_PASSWORD" ]] && missing+=("NEO4J_PASSWORD")

if [[ ${#missing[@]} -gt 0 ]]; then
    log_error "Missing required variables in .env: ${missing[*]}"
    exit 1
fi

# Validate NEO4J_HOST format (should not include protocol)
if [[ "$NEO4J_HOST" == *"://"* ]]; then
    log_error "NEO4J_HOST should not include protocol prefix"
    echo "  Current value: $NEO4J_HOST"
    echo "  Expected format: your-instance.databases.neo4j.io"
    echo "  Remove the 'neo4j+s://' or 'bolt://' prefix from your .env file"
    exit 1
fi

# Validate NEO4J_HOST doesn't include port
if [[ "$NEO4J_HOST" == *":"* ]]; then
    log_warn "NEO4J_HOST contains ':' - make sure it's just the hostname without port"
    echo "  Current value: $NEO4J_HOST"
fi

# Create secret scope (ignore error if already exists)
log_info "Creating secret scope: $SCOPE_NAME"
if databricks secrets create-scope "$SCOPE_NAME" 2>/dev/null; then
    log_info "Secret scope created"
else
    log_warn "Secret scope already exists (or failed to create)"
fi

# Function to set a secret
set_secret() {
    local key=$1
    local value=$2
    log_info "Setting secret: $key"
    echo -n "$value" | databricks secrets put-secret "$SCOPE_NAME" "$key"
}

# Set required secrets
set_secret "host" "$NEO4J_HOST"
set_secret "username" "$NEO4J_USER"
set_secret "password" "$NEO4J_PASSWORD"

# Set optional secrets if provided
if [[ -n "$NEO4J_DATABASE" ]]; then
    set_secret "database" "$NEO4J_DATABASE"
fi

if [[ -n "$NEO4J_PROTOCOL" ]]; then
    set_secret "protocol" "$NEO4J_PROTOCOL"
fi

log_info "Done! Secrets configured in scope: $SCOPE_NAME"
echo ""

# Validate secrets were set correctly
log_info "Validating secrets..."
echo ""

# List secrets to confirm they exist
echo "Secrets in scope '$SCOPE_NAME':"
databricks secrets list-secrets "$SCOPE_NAME"
echo ""

# Validate host value - use the value we already have from .env
log_info "Validating host format..."
log_info "Host value from .env: $NEO4J_HOST"

# Check DNS resolution
if command -v host &> /dev/null; then
    if host "$NEO4J_HOST" &> /dev/null; then
        log_info "Host '$NEO4J_HOST' resolves successfully"
    else
        log_warn "Host '$NEO4J_HOST' did not resolve - verify it's correct"
    fi
elif command -v nslookup &> /dev/null; then
    if nslookup "$NEO4J_HOST" &> /dev/null; then
        log_info "Host '$NEO4J_HOST' resolves successfully"
    else
        log_warn "Host '$NEO4J_HOST' did not resolve - verify it's correct"
    fi
else
    log_info "Host format validated (no DNS lookup tool available)"
fi

echo ""
echo "Use in Databricks notebook:"
echo "  SCOPE_NAME = \"$SCOPE_NAME\"  # <-- Set this to match your scope"
echo "  NEO4J_HOST = dbutils.secrets.get(SCOPE_NAME, \"host\")"
echo "  NEO4J_USER = dbutils.secrets.get(SCOPE_NAME, \"username\")"
echo "  NEO4J_PASSWORD = dbutils.secrets.get(SCOPE_NAME, \"password\")"
echo ""
