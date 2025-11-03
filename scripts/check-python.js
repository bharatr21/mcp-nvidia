#!/usr/bin/env node

import { spawn } from 'child_process';

function checkCommand(command, args = ['--version']) {
  return new Promise((resolve) => {
    const proc = spawn(command, args, { stdio: 'pipe' });
    proc.on('error', () => resolve(false));
    proc.on('close', (code) => resolve(code === 0));
  });
}

async function main() {
  console.log('Checking Python installation...');
  
  const hasPython = await checkCommand('python3', ['--version']);
  
  if (!hasPython) {
    console.warn('\n⚠️  Warning: Python 3 not found.');
    console.warn('The mcp-nvidia package requires Python 3.10 or higher.');
    console.warn('\nPlease install Python 3.10+ from:');
    console.warn('  - https://www.python.org/downloads/');
    console.warn('  - Or use your system package manager');
    console.warn('\nAfter installing Python, run:');
    console.warn('  pip install mcp-nvidia');
    return;
  }
  
  console.log('✓ Python 3 found');
  
  // Check if mcp-nvidia is installed
  const hasMcpNvidia = await checkCommand('python3', ['-m', 'mcp_nvidia', '--version']);
  
  if (!hasMcpNvidia) {
    console.log('\nInstalling mcp-nvidia Python package...');
    console.log('Run: pip install mcp-nvidia');
    console.log('\nOr with uv for faster installation:');
    console.log('  uv pip install mcp-nvidia');
  } else {
    console.log('✓ mcp-nvidia package is installed');
  }
}

main();
