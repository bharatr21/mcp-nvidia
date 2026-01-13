"""CSS styles for MCP-UI components."""

STYLES = """
<style>
  .mcp-nvidia-ui {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1a1a1a;
    max-width: 900px;
    margin: 0 auto;
    padding: 16px;
  }

  .mcp-nvidia-header {
    position: sticky;
    top: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: linear-gradient(90deg, #76b900 0%, #5a9a00 100%);
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .mcp-nvidia-logo {
    font-weight: 700;
    font-size: 18px;
    color: white;
    letter-spacing: -0.5px;
  }

  .mcp-nvidia-title {
    font-size: 14px;
    color: rgba(255, 255, 255, 0.9);
  }

  .mcp-nvidia-search-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
  }

  .mcp-nvidia-search-input {
    flex: 1;
    padding: 10px 14px;
    border: 1px solid #ddd;
    border-radius: 6px;
    font-size: 14px;
  }

  .mcp-nvidia-search-input:focus {
    outline: none;
    border-color: #76b900;
    box-shadow: 0 0 0 2px rgba(118, 185, 0, 0.2);
  }

  .mcp-nvidia-filter-panel {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px 16px;
    background: #f8f9fa;
    border-radius: 8px;
    margin-bottom: 16px;
    align-items: center;
  }

  .mcp-nvidia-filter-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .mcp-nvidia-filter-label {
    font-size: 12px;
    font-weight: 500;
    color: #666;
  }

  .mcp-nvidia-select {
    padding: 6px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 13px;
    background: white;
    cursor: pointer;
  }

  .mcp-nvidia-select:focus {
    outline: none;
    border-color: #76b900;
  }

  .mcp-nvidia-range-container {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .mcp-nvidia-range {
    width: 120px;
    accent-color: #76b900;
  }

  .mcp-nvidia-range-value {
    font-size: 13px;
    font-weight: 500;
    min-width: 32px;
    color: #76b900;
  }

  .mcp-nvidia-results-count {
    font-size: 13px;
    color: #666;
    margin-left: auto;
  }

  .mcp-nvidia-results-time {
    font-size: 12px;
    color: #999;
  }

  .mcp-nvidia-results-container {
    margin-bottom: 20px;
  }

  .mcp-nvidia-result-card {
    background: white;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s ease;
  }

  .mcp-nvidia-result-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  }

  .mcp-nvidia-result-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 8px;
  }

  .mcp-nvidia-relevance-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 40px;
    height: 24px;
    padding: 0 8px;
    background: linear-gradient(90deg, #76b900 var(--score), #e8e8e8 var(--score));
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    color: #333;
  }

  .mcp-nvidia-result-title {
    flex: 1;
    font-size: 16px;
    font-weight: 600;
    color: #1a1a1a;
    text-decoration: none;
    line-height: 1.4;
  }

  .mcp-nvidia-result-title:hover {
    color: #76b900;
    text-decoration: underline;
  }

  .mcp-nvidia-result-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 10px;
    font-size: 12px;
    color: #666;
  }

  .mcp-nvidia-domain-tag {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    background: #f0f0f0;
    border-radius: 4px;
    font-size: 11px;
    color: #555;
  }

  .mcp-nvidia-content-type {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    background: #e8f5e9;
    border-radius: 4px;
    font-size: 11px;
    color: #2e7d32;
  }

  .mcp-nvidia-date {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: #888;
  }

  .mcp-nvidia-result-snippet {
    font-size: 14px;
    line-height: 1.6;
    color: #444;
    margin-bottom: 10px;
  }

  .mcp-nvidia-keywords {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
  }

  .mcp-nvidia-keyword {
    padding: 2px 8px;
    background: #fff3e0;
    border-radius: 12px;
    font-size: 11px;
    color: #e65100;
  }

  .mcp-nvidia-result-actions {
    display: flex;
    gap: 8px;
  }

  .mcp-nvidia-btn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .mcp-nvidia-btn-primary {
    background: #76b900;
    color: white;
    border: none;
  }

  .mcp-nvidia-btn-primary:hover {
    background: #5a9a00;
  }

  .mcp-nvidia-btn-secondary {
    background: white;
    color: #333;
    border: 1px solid #ddd;
  }

  .mcp-nvidia-btn-secondary:hover {
    background: #f5f5f5;
  }

  .mcp-nvidia-citations {
    border-top: 1px solid #e5e5e5;
    padding-top: 16px;
    margin-top: 20px;
  }

  .mcp-nvidia-citations-title {
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #333;
  }

  .mcp-nvidia-citation-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .mcp-nvidia-citation {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 12px;
    background: #f8f9fa;
    border-radius: 6px;
    font-size: 12px;
    color: #555;
  }

  .mcp-nvidia-citation-number {
    font-weight: 600;
    color: #76b900;
    min-width: 24px;
  }

  .mcp-nvidia-citation-link {
    color: #76b900;
    text-decoration: none;
  }

 .mcp-nvidia-citation-link:hover {
    text-decoration: underline;
  }

  .mcp-nvidia-content-type-tabs {
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    padding: 4px;
    background: #f0f0f0;
    border-radius: 8px;
  }

  .mcp-nvidia-tab {
    flex: 1;
    padding: 8px 16px;
    border: none;
    background: transparent;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    color: #666;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .mcp-nvidia-tab.active {
    background: white;
    color: #333;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .mcp-nvidia-tab:hover:not(.active) {
    color: #333;
  }

  .mcp-nvidia-content-card {
    display: flex;
    gap: 16px;
    padding: 16px;
    background: white;
    border: 1px solid #e5e5e5;
    border-radius: 8px;
    margin-bottom: 12px;
    transition: box-shadow 0.2s ease;
  }

  .mcp-nvidia-content-card:hover {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  }

  .mcp-nvidia-content-thumbnail {
    width: 120px;
    height: 80px;
    background: #f0f0f0;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    flex-shrink: 0;
  }

  .mcp-nvidia-content-info {
    flex: 1;
    min-width: 0;
  }

  .mcp-nvidia-content-title {
    font-size: 15px;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 6px;
    text-decoration: none;
  }

  .mcp-nvidia-content-title:hover {
    color: #76b900;
  }

  .mcp-nvidia-content-domain {
    font-size: 12px;
    color: #888;
    margin-bottom: 8px;
  }

  .mcp-nvidia-content-score {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    background: #e8f5e9;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 500;
    color: #2e7d32;
  }

  .mcp-nvidia-content-snippet {
    font-size: 13px;
    line-height: 1.5;
    color: #555;
    margin-top: 8px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .mcp-nvidia-empty-state {
    text-align: center;
    padding: 48px 24px;
    color: #666;
  }

  .mcp-nvidia-empty-icon {
    font-size: 48px;
    margin-bottom: 16px;
  }

  .mcp-nvidia-empty-title {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin-bottom: 8px;
  }

  .mcp-nvidia-empty-message {
    font-size: 14px;
    line-height: 1.6;
  }

  .mcp-nvidia-error {
    padding: 16px;
    background: #ffebee;
    border-radius: 8px;
    color: #c62828;
  }

  .mcp-nvidia-warning {
    padding: 12px 16px;
    background: #fff8e1;
    border-radius: 6px;
    font-size: 13px;
    color: #f57f17;
    margin-bottom: 16px;
  }
</style>
"""
