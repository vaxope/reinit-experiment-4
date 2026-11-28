#!/bin/bash
# scripts/deltaai/install_code_server.sh: download and unpack the pinned code-server, on a DeltaAI LOGIN node.
#
# Run once (login nodes have internet; compute nodes may not):  bash scripts/deltaai/install_code_server.sh
# What : downloads configs/code_server.toml's `url`, refuses it unless its SHA-256 equals `sha256`,
#        unpacks it into $CODE_SERVER_HOME (default ~/.gar_code_server, outside the repo), and creates
#        a random password file $CODE_SERVER_HOME/password (mode 600) if none exists. Nothing is
#        installed system-wide and no package manager is used; undo = rm -rf ~/.gar_code_server.
# Used by: scripts/deltaai/hold_node.sbatch (starts it on the compute node).
set -euo pipefail
cd "$(dirname "$0")/../.."
CS_HOME="${CODE_SERVER_HOME:-$HOME/.gar_code_server}"
get() { sed -n "s/^$1 = \"\(.*\)\"\$/\1/p" configs/code_server.toml | head -n 1; }
VERSION="$(get version)"; URL="$(get url)"; SHA="$(get sha256)"
DEST="$CS_HOME/code-server-$VERSION-linux-arm64"
if [ -x "$DEST/bin/code-server" ]; then
  echo "[code-server] $VERSION already installed in $DEST"
else
  mkdir -p "$CS_HOME"
  TMP="$CS_HOME/download_$VERSION.tar.gz"
  echo "[code-server] downloading $URL"
  curl -fL --retry 3 -o "$TMP" "$URL"
  GOT="$(sha256sum "$TMP" | cut -d' ' -f1)"
  if [ "$GOT" != "$SHA" ]; then
    rm -f "$TMP"   # this script's own download
    echo "[code-server] SHA-256 $GOT != pinned $SHA: refusing (configs/code_server.toml)"; exit 1
  fi
  tar -xzf "$TMP" -C "$CS_HOME"
  rm -f "$TMP"
  echo "[code-server] $VERSION installed in $DEST (sha256 checked)"
fi
if [ ! -f "$CS_HOME/password" ]; then
  ( umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=' > "$CS_HOME/password" )
  echo "[code-server] password written to $CS_HOME/password (only you can read it)"
fi
"$DEST/bin/code-server" --version | head -n 1
echo "[code-server] next: scripts/deltaai/submit.sh hold   (holds a 4-GPU node; charged while it runs)"
