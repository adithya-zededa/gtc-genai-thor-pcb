#!/bin/bash
# ZEDEDA Camera Monitoring Agent - Setup and Run Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_SCRIPT="$SCRIPT_DIR/camera_agent.py"
CONFIG_FILE="$SCRIPT_DIR/camera_config.yaml"
ENV_FILE="$SCRIPT_DIR/.env"
LOG_FILE="$SCRIPT_DIR/camera_agent.log"

# Default environment variables
DEFAULT_OLLAMA_URL="http://localhost:11434"
DEFAULT_VISION_MODEL="llava:7b"
DEFAULT_CAMERA_INDEX="0"
DEFAULT_CAPTURE_INTERVAL="5"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if Ollama is running
check_ollama() {
    log "Checking Ollama service..."
    
    local ollama_url="${OLLAMA_URL:-$DEFAULT_OLLAMA_URL}"
    
    if curl -s -f "$ollama_url/api/version" >/dev/null 2>&1; then
        success "Ollama is running at $ollama_url"
        
        # Check for GPU acceleration
        if docker ps --format "table {{.Names}}\t{{.Image}}" | grep ollama >/dev/null 2>&1; then
            local container_name=$(docker ps --format "{{.Names}}" | grep ollama)
            local runtime=$(docker inspect "$container_name" | grep -i "\"Runtime\":" | cut -d'"' -f4)
            if [[ "$runtime" == "nvidia" ]]; then
                success "Ollama is using GPU acceleration (nvidia runtime)"
            else
                warning "Ollama is using CPU only ($runtime runtime)"
            fi
        fi
        
        return 0
    else
        error "Ollama is not accessible at $ollama_url"
        return 1
    fi
}

# Function to check camera access
check_camera() {
    log "Checking camera access..."
    
    local camera_index="${CAMERA_INDEX:-$DEFAULT_CAMERA_INDEX}"
    local camera_device="/dev/video$camera_index"
    
    if [[ ! -e "$camera_device" ]]; then
        error "Camera device $camera_device not found"
        return 1
    fi
    
    if [[ ! -r "$camera_device" ]]; then
        error "No read permission for $camera_device"
        warning "Try running: sudo chmod 666 $camera_device"
        return 1
    fi
    
    success "Camera device $camera_device is accessible"
    return 0
}

# Function to check Python dependencies
check_dependencies() {
    log "Checking Python dependencies..."
    
    local required_packages=("cv2" "yaml" "requests" "dotenv")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" >/dev/null 2>&1; then
            missing_packages+=("$package")
        fi
    done
    
    if [[ ${#missing_packages[@]} -gt 0 ]]; then
        error "Missing Python packages: ${missing_packages[*]}"
        log "Install with: pip3 install opencv-python pyyaml requests python-dotenv"
        return 1
    fi
    
    success "All Python dependencies are installed"
    return 0
}

# Function to create environment file if it doesn't exist
create_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log "Creating environment file..."
        cat > "$ENV_FILE" << EOF
# ZEDEDA Camera Agent Environment Configuration
OLLAMA_URL=$DEFAULT_OLLAMA_URL
VISION_MODEL=$DEFAULT_VISION_MODEL
CAMERA_INDEX=$DEFAULT_CAMERA_INDEX
CAPTURE_INTERVAL=$DEFAULT_CAPTURE_INTERVAL
LOG_LEVEL=INFO

# Email configuration (update these)
EMAIL_USER=your-email@gmail.com
EMAIL_PASS=your-app-password

# Optional: Slack webhook
SLACK_WEBHOOK_URL=
EOF
        success "Environment file created at $ENV_FILE"
        warning "Please update the email configuration in $ENV_FILE"
    else
        log "Environment file already exists: $ENV_FILE"
    fi
}

# Function to load environment variables
load_env() {
    if [[ -f "$ENV_FILE" ]]; then
        log "Loading environment variables from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    fi
}

# Function to run system test
run_test() {
    log "Running camera agent system test..."
    
    if ! python3 "$AGENT_SCRIPT" --config "$CONFIG_FILE" --test; then
        error "System test failed"
        return 1
    fi
    
    success "System test passed!"
    return 0
}

# Function to start the camera agent
start_agent() {
    log "Starting ZEDEDA Camera Monitoring Agent..."
    
    if [[ -f "$LOG_FILE" ]]; then
        mv "$LOG_FILE" "${LOG_FILE}.bak"
    fi
    
    log "Agent will log to: $LOG_FILE"
    log "Press Ctrl+C to stop the agent"
    log "Starting in 3 seconds..."
    sleep 3
    
    exec python3 "$AGENT_SCRIPT" --config "$CONFIG_FILE" --log-level "${LOG_LEVEL:-INFO}"
}

# Function to install systemd service
install_service() {
    log "Installing systemd service..."
    
    local service_file="$SCRIPT_DIR/zededa-camera-agent.service"
    local system_service="/etc/systemd/system/zededa-camera-agent.service"
    
    if [[ ! -f "$service_file" ]]; then
        error "Service file not found: $service_file"
        return 1
    fi
    
    sudo cp "$service_file" "$system_service"
    sudo systemctl daemon-reload
    sudo systemctl enable zededa-camera-agent
    
    success "Systemd service installed and enabled"
    log "Start with: sudo systemctl start zededa-camera-agent"
    log "Check status: sudo systemctl status zededa-camera-agent"
    log "View logs: sudo journalctl -u zededa-camera-agent -f"
}

# Function to show usage
show_usage() {
    cat << EOF
ZEDEDA Camera Monitoring Agent - Setup and Run Script

Usage: $0 [COMMAND]

Commands:
    test        Run system test to verify setup
    start       Start the camera monitoring agent
    service     Install systemd service
    docker      Build and run Docker containers
    check       Check system requirements
    setup       Create environment configuration
    help        Show this help message

Examples:
    $0 setup       # Create environment configuration
    $0 check       # Check system requirements
    $0 test        # Run system test
    $0 start       # Start monitoring agent
    $0 service     # Install as systemd service

Environment Variables:
    OLLAMA_URL           - Ollama server URL (default: $DEFAULT_OLLAMA_URL)
    VISION_MODEL         - Vision model name (default: $DEFAULT_VISION_MODEL)
    CAMERA_INDEX         - Camera device index (default: $DEFAULT_CAMERA_INDEX)
    CAPTURE_INTERVAL     - Seconds between captures (default: $DEFAULT_CAPTURE_INTERVAL)
    EMAIL_USER           - Email username for alerts
    EMAIL_PASS           - Email password for alerts
    LOG_LEVEL            - Logging level (DEBUG, INFO, WARNING, ERROR)

EOF
}

# Function to run system checks
run_checks() {
    log "Running system checks..."
    
    local checks_passed=0
    local total_checks=4
    
    # Check Python
    if command_exists python3; then
        success "Python 3 is installed"
        ((checks_passed++))
    else
        error "Python 3 is not installed"
    fi
    
    # Check dependencies
    if check_dependencies; then
        ((checks_passed++))
    fi
    
    # Check Ollama
    if check_ollama; then
        ((checks_passed++))
    fi
    
    # Check camera
    if check_camera; then
        ((checks_passed++))
    fi
    
    log "System checks: $checks_passed/$total_checks passed"
    
    if [[ $checks_passed -eq $total_checks ]]; then
        success "All system checks passed!"
        return 0
    else
        error "Some system checks failed"
        return 1
    fi
}

# Function to build and run Docker containers
run_docker() {
    log "Building and running Docker containers..."
    
    if ! command_exists docker; then
        error "Docker is not installed"
        return 1
    fi
    
    if ! command_exists docker-compose; then
        error "Docker Compose is not installed"
        return 1
    fi
    
    # Build containers
    log "Building containers..."
    docker-compose build
    
    # Start services
    log "Starting services..."
    docker-compose up -d
    
    # Show status
    log "Container status:"
    docker-compose ps
    
    success "Docker containers are running"
    log "View logs with: docker-compose logs -f camera-agent"
    log "Stop with: docker-compose down"
}

# Main script logic
main() {
    case "${1:-}" in
        "test")
            load_env
            run_test
            ;;
        "start")
            load_env
            if run_checks; then
                start_agent
            else
                error "System checks failed. Please fix issues before starting."
                exit 1
            fi
            ;;
        "service")
            install_service
            ;;
        "docker")
            run_docker
            ;;
        "check")
            load_env
            run_checks
            ;;
        "setup")
            create_env_file
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        "")
            log "ZEDEDA Camera Monitoring Agent"
            log "Run '$0 help' for usage information"
            log "Run '$0 setup' to get started"
            ;;
        *)
            error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"