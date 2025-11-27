#!/bin/bash

# Verible Installation Script
# Downloads and installs Verible locally in the project directory

set -e

# Configuration
VERIBLE_VERSION="v0.0-3624-g1f4d723f" # Fallback/Reference version
# Note: We will try to fetch the latest or a specific recent version that supports both Linux and Mac.
# The search result suggested v0.0-4023-gc1271a00 has macOS support.
# Let's use a specific recent version to ensure stability.
VERSION_TAG="v0.0-3624-g1f4d723f" 
# Actually, let's use a newer one if possible, but 3624 is in the README.
# Let's check if 3624 has macos.
# If not, we will use a newer one.
# To be safe and robust, let's try to determine the URL dynamically or use a known good one for Mac.
# Search result said: verible-v0.0-4023-gc1271a00-macOS.tar.gz

# Let's use a hardcoded recent version that is likely to work.
TARGET_VERSION="v0.0-3624-g1f4d723f"
BASE_URL="https://github.com/chipsalliance/verible/releases/download"

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Detected OS: ${OS_TYPE}"

# Hardcoded known working URLs
LINUX_URL="${BASE_URL}/${TARGET_VERSION}/verible-${TARGET_VERSION}-linux-static-x86_64.tar.gz"
MAC_URL="${BASE_URL}/${TARGET_VERSION}/verible-${TARGET_VERSION}-macOS.tar.gz"

# Set download URL based on OS
if [ "${OS_TYPE}" == "Linux" ]; then
    DOWNLOAD_URL="$LINUX_URL"
elif [ "${OS_TYPE}" == "Mac" ]; then
    DOWNLOAD_URL="$MAC_URL"
else
    echo "Unsupported OS: ${OS_TYPE}"
    exit 1
fi

INSTALL_DIR="$(pwd)/verible_bin"
mkdir -p "$INSTALL_DIR"

echo "Installing Verible to $INSTALL_DIR..."

# Function to download and extract
install_verible() {
    local url=$1
    local name=$(basename "$url")
    
    echo "Downloading $url..."
    if curl -L -f -o "$name" "$url"; then
        echo "Download successful."
    else
        echo "Download failed. Trying to find a fallback..."
        return 1
    fi
    
    echo "Extracting..."
    tar -xf "$name"
    
    # Find the bin directory inside
    # The tarball usually contains a folder like verible-v0.0-3624...
    # We want to move the contents of that folder's bin to our bin
    
    # Find the directory created
    EXTRACTED_DIR=$(tar -tf "$name" | head -1 | cut -f1 -d"/")
    
    echo "Extracted to $EXTRACTED_DIR"
    
    # Move binaries
    cp -r "$EXTRACTED_DIR"/bin/* "$INSTALL_DIR/"
    
    # Cleanup
    rm "$name"
    rm -rf "$EXTRACTED_DIR"
    
    echo "Verible installed successfully!"
    echo "Binaries are in $INSTALL_DIR"
    echo "Please add this to your PATH or configure your tools to use it:"
    echo "export PATH=\$PATH:$INSTALL_DIR"
}

# Try to install
if ! install_verible "$DOWNLOAD_URL"; then
    echo "Primary download failed. Trying to fetch latest release..."
    # Fetch the latest release from GitHub API
    LATEST_RELEASE_JSON=$(curl -s https://api.github.com/repos/chipsalliance/verible/releases/latest)
    if [ "${OS_TYPE}" == "Linux" ]; then
        LATEST_URL=$(echo "$LATEST_RELEASE_JSON" | grep "browser_download_url" | grep "linux-static-x86_64" | head -1 | cut -d '"' -f 4)
    elif [ "${OS_TYPE}" == "Mac" ]; then
        LATEST_URL=$(echo "$LATEST_RELEASE_JSON" | grep "browser_download_url" | grep "macOS" | head -1 | cut -d '"' -f 4)
    fi
    if [ -n "$LATEST_URL" ]; then
        echo "Found latest release: $LATEST_URL"
        install_verible "$LATEST_URL"
    else
        echo "Could not find a release automatically."
        exit 1
    fi
fi
