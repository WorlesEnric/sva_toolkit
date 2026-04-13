#!/bin/bash

# Yosys (OSS CAD Suite) Installation Script
# Downloads and installs the latest OSS CAD Suite from GitHub
# Reference: https://github.com/YosysHQ/oss-cad-suite-build

set -e

# Configuration
REPO="YosysHQ/oss-cad-suite-build"

# Detect OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${OS}" in
    Linux*)     OS_TYPE="linux";;
    Darwin*)    OS_TYPE="darwin";;
    *)          echo "Unsupported OS: ${OS}"; exit 1;;
esac

case "${ARCH}" in
    x86_64)     ARCH_TYPE="x64";;
    arm64|aarch64) ARCH_TYPE="arm64";;
    *)          echo "Unsupported Architecture: ${ARCH}"; exit 1;;
esac

echo "Detected System: ${OS_TYPE} ${ARCH_TYPE}"

# Fetch latest release info
echo "Fetching latest release info from GitHub..."
# Use curl to get the latest release JSON
LATEST_RELEASE_JSON=$(curl -s "https://api.github.com/repos/${REPO}/releases/latest")

# Extract the download URL for the matching asset
# We are looking for a file pattern like: oss-cad-suite-darwin-arm64-*.tgz
SEARCH_PATTERN="oss-cad-suite-${OS_TYPE}-${ARCH_TYPE}"
DOWNLOAD_URL=$(echo "$LATEST_RELEASE_JSON" | grep "browser_download_url" | grep "${SEARCH_PATTERN}" | grep ".tgz" | head -1 | cut -d '"' -f 4)

if [ -z "$DOWNLOAD_URL" ]; then
    echo "Error: Could not find a release asset matching '${SEARCH_PATTERN}' in the latest release."
    exit 1
fi

FILE_NAME=$(basename "$DOWNLOAD_URL")
INSTALL_DIR="$(pwd)/oss-cad-suite"

echo "Found latest release: $FILE_NAME"
echo "URL: $DOWNLOAD_URL"

# Download
if [ -f "$FILE_NAME" ]; then
    echo "Archive $FILE_NAME already exists. Skipping download."
else
    echo "Downloading..."
    curl -L -f -o "$FILE_NAME" "$DOWNLOAD_URL"
fi

# Extract
echo "Extracting..."
# Clean up previous install
if [ -d "oss-cad-suite" ]; then
    echo "Removing existing oss-cad-suite directory..."
    rm -rf oss-cad-suite
fi

tar -xf "$FILE_NAME"

# macOS Specific Setup
if [ "${OS_TYPE}" == "darwin" ]; then
    echo "Running macOS setup..."
    
    # 1. Remove quarantine attribute from the archive (if it persists)
    if command -v xattr >/dev/null 2>&1; then
        xattr -d com.apple.quarantine "$FILE_NAME" || true
    fi

    # 2. Run the activation script in the extracted directory
    if [ -f "${INSTALL_DIR}/activate" ]; then
        echo "Executing activate script to initialize environment..."
        # Execute in a subshell
        (cd "${INSTALL_DIR}" && ./activate)
    fi
fi

echo "----------------------------------------------------------------"
echo "Yosys (OSS CAD Suite) installed successfully!"
echo "Installation Directory: ${INSTALL_DIR}"
echo ""
echo "To use the tools, set up your environment:"
echo ""
echo "  export PATH=\"${INSTALL_DIR}/bin:\$PATH\""
echo ""
echo "Or source the environment file:"
echo ""
echo "  source \"${INSTALL_DIR}/environment\""
echo "----------------------------------------------------------------"

