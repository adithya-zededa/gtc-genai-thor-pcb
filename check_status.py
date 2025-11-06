#!/usr/bin/env python3
"""
Simple status checker for the ZEDEDA AI Agent API service.
Shows clear status indicators for all components.
"""

import requests
import json
from datetime import datetime
import sys

def colorize(text, color):
    """Add color to terminal output"""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def get_status_icon(status):
    """Get icon based on status"""
    if status in ['healthy', 'ok', 'ready']:
        return colorize('✅', 'green')
    elif status in ['warning']:
        return colorize('⚠️ ', 'yellow')
    elif status in ['error', 'degraded']:
        return colorize('❌', 'red')
    else:
        return colorize('❓', 'blue')

def check_service_status(base_url="http://localhost:8080"):
    """Check the status of all service endpoints"""
    
    print(f"\n{colorize('🔍 ZEDEDA AI Agent Status Check', 'blue')}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Service URL: {base_url}")
    print("=" * 60)
    
    try:
        # Check root endpoint
        print(f"\n{colorize('📋 Service Info:', 'blue')}")
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            print(f"  {get_status_icon(status)} Service: {status}")
            print(f"  🤖 Model: {data.get('model', 'unknown')}")
            print(f"  🔗 Ollama URL: {data.get('ollama_url', 'unknown')}")
        else:
            print(f"  ❌ Service unreachable (HTTP {response.status_code})")
            return False
            
    except requests.RequestException as e:
        print(f"  ❌ Service unreachable: {e}")
        return False

    try:
        # Check health endpoint
        print(f"\n{colorize('🏥 Health Check:', 'blue')}")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'unknown')
            print(f"  {get_status_icon(status)} Overall Health: {status}")
        else:
            print(f"  ❌ Health check failed (HTTP {response.status_code})")
            
    except requests.RequestException as e:
        print(f"  ❌ Health check failed: {e}")

    try:
        # Check system health
        print(f"\n{colorize('🖥️  System Health:', 'blue')}")
        response = requests.get(f"{base_url}/api/system-health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            overall = data.get('overall', 'unknown')
            print(f"  {get_status_icon(overall)} Overall: {overall}")
            
            components = data.get('components', {})
            for name, info in components.items():
                status = info.get('status', 'unknown')
                detail = info.get('detail', '')
                print(f"  {get_status_icon(status)} {name.title()}: {status} ({detail})")
                
        else:
            print(f"  ❌ System health check failed (HTTP {response.status_code})")
            
    except requests.RequestException as e:
        print(f"  ❌ System health check failed: {e}")

    # Test API endpoints
    print(f"\n{colorize('🧪 API Endpoint Tests:', 'blue')}")
    
    try:
        # Test simple generation
        response = requests.post(
            f"{base_url}/api/generate",
            json={"prompt": "Hello", "stream": False},
            timeout=30
        )
        if response.status_code == 200:
            print(f"  ✅ Text generation: Working")
        else:
            print(f"  ❌ Text generation: Failed (HTTP {response.status_code})")
            
    except requests.RequestException as e:
        print(f"  ❌ Text generation: Failed ({e})")

    try:
        # Test agent endpoint
        response = requests.post(
            f"{base_url}/api/agent/run",
            json={"subject": "Status Test", "body": "Testing agent"},
            timeout=30
        )
        if response.status_code == 200:
            print(f"  ✅ Agent workflow: Working")
        else:
            print(f"  ❌ Agent workflow: Failed (HTTP {response.status_code})")
            
    except requests.RequestException as e:
        print(f"  ❌ Agent workflow: Failed ({e})")

    print("\n" + "=" * 60)
    print(f"{colorize('✨ Status check complete!', 'green')}")
    return True

if __name__ == "__main__":
    # Default to port-forwarded service, but allow override
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    if not check_service_status(base_url):
        sys.exit(1)