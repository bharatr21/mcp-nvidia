#!/usr/bin/env node

import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Check if uv is available, otherwise fall back to python -m
function findPythonCommand() {
  return new Promise((resolve) => {
    const uvCheck = spawn('uv', ['--version']);
    
    uvCheck.on('error', () => {
      // uv not available, use python
      resolve(['python3', '-m', 'mcp_nvidia.server']);
    });
    
    uvCheck.on('close', (code) => {
      if (code === 0) {
        // uv is available
        resolve(['uvx', 'mcp-nvidia']);
      } else {
        // uv not available, use python
        resolve(['python3', '-m', 'mcp_nvidia.server']);
      }
    });
  });
}

async function main() {
  try {
    const [command, ...args] = await findPythonCommand();
    
    const child = spawn(command, args, {
      stdio: 'inherit',
      env: {
        ...process.env,
        // Pass through any npm config as environment variables
      }
    });

    child.on('error', (error) => {
      console.error('Failed to start mcp-nvidia:', error.message);
      console.error('\nPlease ensure Python 3.10+ and mcp-nvidia package are installed:');
      console.error('  pip install mcp-nvidia');
      console.error('\nOr install with uv:');
      console.error('  uv pip install mcp-nvidia');
      process.exit(1);
    });

    child.on('exit', (code) => {
      process.exit(code || 0);
    });
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
