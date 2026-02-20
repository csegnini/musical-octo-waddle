"""
Documentation Engine Module

This module provides comprehensive documentation generation capabilities for workflows,
including HTML reports, markdown documentation, and summary generation.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path

from .workflow_manager import WorkflowManager

logger = logging.getLogger(__name__)


class DocumentationEngine:
    """
    Comprehensive documentation generation engine for workflows.
    
    This class generates various types of documentation including HTML reports,
    markdown files, and structured summaries.
    """
    
    def __init__(self, workflow_manager: WorkflowManager):
        """
        Initialize documentation engine.
        
        Args:
            workflow_manager: Workflow manager instance
        """
        self.workflow_manager = workflow_manager
        self.workflow_id = workflow_manager.workflow_id
        
        logger.info(f"Initialized documentation engine for workflow {self.workflow_id}")
    
    def generate_html_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive HTML report.
        
        Args:
            output_path: Path for the HTML file
            
        Returns:
            Path to generated HTML file
        """
        if not output_path:
            output_path = os.path.join(
                self.workflow_manager.output_folder, 
                "reports", 
                f"workflow_report_{self.workflow_id[:8]}.html"
            )
        
        # Get workflow data
        summary = self.workflow_manager.get_workflow_summary()
        documentation = self.workflow_manager.generate_full_documentation()
        
        html_content = self._generate_html_content(summary, documentation)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        return output_path
    
    def _generate_html_content(self, summary: Dict[str, Any], documentation: Dict[str, Any]) -> str:
        """Generate HTML content for the report."""
        
        # Get execution statistics
        execution_summary = summary.get('execution_summary', {})
        coordinator_stats = summary.get('agent_coordination_summary', {})
        inventory_stats = summary.get('data_inventory_summary', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Workflow Report - {self.workflow_id[:8]}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #495057;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #6c757d;
            margin-top: 25px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            border-left: 4px solid #007bff;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-completed {{ background: #d4edda; color: #155724; }}
        .status-in-progress {{ background: #d1ecf1; color: #0c5460; }}
        .status-failed {{ background: #f8d7da; color: #721c24; }}
        .status-pending {{ background: #fff3cd; color: #856404; }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .table th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }}
        .table tr:hover {{
            background-color: #f5f5f5;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s ease;
        }}
        .json-section {{
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }}
        .footer {{
            text-align: center;
            color: #6c757d;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Workflow Report</h1>
        <p>Workflow ID: {self.workflow_id}</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>

    <div class="section">
        <h2>📊 Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{execution_summary.get('progress_percentage', 0):.1f}%</div>
                <div class="stat-label">Progress</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_summary.get('completed_steps', 0)}</div>
                <div class="stat-label">Completed Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{coordinator_stats.get('total_agents', 0)}</div>
                <div class="stat-label">Total Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{inventory_stats.get('total_data_references', 0)}</div>
                <div class="stat-label">Data Files</div>
            </div>
        </div>
        
        <h3>Current Status</h3>
        <p>
            <span class="status-badge status-{summary.get('current_status', 'unknown').lower()}">
                {summary.get('current_status', 'Unknown')}
            </span>
        </p>
        
        <h3>Progress Overview</h3>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {execution_summary.get('progress_percentage', 0)}%"></div>
        </div>
        
        <p><strong>User Query:</strong> {summary.get('user_query', 'N/A')}</p>
        <p><strong>Output Folder:</strong> {summary.get('output_folder', 'N/A')}</p>
    </div>

    <div class="section">
        <h2>⚙️ Execution Details</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{execution_summary.get('total_steps', 0)}</div>
                <div class="stat-label">Total Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_summary.get('failed_steps', 0)}</div>
                <div class="stat-label">Failed Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_summary.get('running_steps', 0)}</div>
                <div class="stat-label">Running Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{execution_summary.get('total_duration_seconds', 0):.1f}s</div>
                <div class="stat-label">Total Duration</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🤖 Agent Coordination</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{coordinator_stats.get('available_agents', 0)}</div>
                <div class="stat-label">Available Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{coordinator_stats.get('busy_agents', 0)}</div>
                <div class="stat-label">Busy Agents</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{coordinator_stats.get('total_work_orders', 0)}</div>
                <div class="stat-label">Total Work Orders</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{coordinator_stats.get('success_rate', 0):.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📁 Data Inventory</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{inventory_stats.get('total_size_bytes', 0) / (1024*1024):.1f} MB</div>
                <div class="stat-label">Total Data Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{inventory_stats.get('total_record_count', 0):,}</div>
                <div class="stat-label">Total Records</div>
            </div>
        </div>
        
        <h3>Data Types Distribution</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Data Type</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Add data types table
        data_types = inventory_stats.get('data_types', {})
        for data_type, count in data_types.items():
            html += f"""
                <tr>
                    <td>{data_type.replace('_', ' ').title()}</td>
                    <td>{count}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>🎯 Objectives & KPIs</h2>
"""
        
        objectives = documentation.get('workflow_definition', {}).get('general_requirements', {}).get('objectives', [])
        kpis = documentation.get('workflow_definition', {}).get('general_requirements', {}).get('kpis', [])
        
        if objectives:
            html += """
        <h3>Objectives</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Target</th>
                    <th>Status</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
"""
            for obj in objectives:
                html += f"""
                <tr>
                    <td>{obj.get('name', 'N/A')}</td>
                    <td>{obj.get('target', 'N/A')}</td>
                    <td><span class="status-badge status-{obj.get('status', 'pending').lower().replace('_', '-')}">{obj.get('status', 'Pending')}</span></td>
                    <td>{obj.get('description', 'N/A')}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""
        
        if kpis:
            html += """
        <h3>Key Performance Indicators</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Target</th>
                    <th>Current Value</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
            for kpi in kpis:
                html += f"""
                <tr>
                    <td>{kpi.get('name', 'N/A')}</td>
                    <td>{kpi.get('target', 'N/A')}</td>
                    <td>{kpi.get('current_value', 'N/A')} {kpi.get('measurement_unit', '')}</td>
                    <td><span class="status-badge status-{kpi.get('status', 'pending').lower().replace('_', '-')}">{kpi.get('status', 'Pending')}</span></td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""
        
        if not objectives and not kpis:
            html += "<p><em>No objectives or KPIs defined yet.</em></p>"
        
        html += """
    </div>

    <div class="section">
        <h2>📋 Complete Documentation (JSON)</h2>
        <div class="json-section">
"""
        html += json.dumps(documentation, indent=2, default=str).replace('<', '&lt;').replace('>', '&gt;')
        
        html += """
        </div>
    </div>

    <div class="footer">
        <p>Generated by Workflow Management System | """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC') + """</p>
    </div>

</body>
</html>
"""
        
        return html
    
    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a markdown report.
        
        Args:
            output_path: Path for the markdown file
            
        Returns:
            Path to generated markdown file
        """
        if not output_path:
            output_path = os.path.join(
                self.workflow_manager.output_folder,
                "reports",
                f"workflow_report_{self.workflow_id[:8]}.md"
            )
        
        # Get workflow data
        summary = self.workflow_manager.get_workflow_summary()
        documentation = self.workflow_manager.generate_full_documentation()
        
        markdown_content = self._generate_markdown_content(summary, documentation)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Generated markdown report: {output_path}")
        return output_path
    
    def _generate_markdown_content(self, summary: Dict[str, Any], documentation: Dict[str, Any]) -> str:
        """Generate markdown content for the report."""
        
        execution_summary = summary.get('execution_summary', {})
        coordinator_stats = summary.get('agent_coordination_summary', {})
        inventory_stats = summary.get('data_inventory_summary', {})
        
        markdown = f"""# 🚀 Workflow Report

**Workflow ID:** {self.workflow_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| Progress | {execution_summary.get('progress_percentage', 0):.1f}% |
| Completed Steps | {execution_summary.get('completed_steps', 0)} |
| Total Agents | {coordinator_stats.get('total_agents', 0)} |
| Data Files | {inventory_stats.get('total_data_references', 0)} |

**Current Status:** {summary.get('current_status', 'Unknown')}

**User Query:** {summary.get('user_query', 'N/A')}

**Output Folder:** {summary.get('output_folder', 'N/A')}

## ⚙️ Execution Details

| Metric | Value |
|--------|-------|
| Total Steps | {execution_summary.get('total_steps', 0)} |
| Completed Steps | {execution_summary.get('completed_steps', 0)} |
| Failed Steps | {execution_summary.get('failed_steps', 0)} |
| Running Steps | {execution_summary.get('running_steps', 0)} |
| Total Duration | {execution_summary.get('total_duration_seconds', 0):.1f}s |

## 🤖 Agent Coordination

| Metric | Value |
|--------|-------|
| Available Agents | {coordinator_stats.get('available_agents', 0)} |
| Busy Agents | {coordinator_stats.get('busy_agents', 0)} |
| Total Work Orders | {coordinator_stats.get('total_work_orders', 0)} |
| Success Rate | {coordinator_stats.get('success_rate', 0):.1f}% |

## 📁 Data Inventory

| Metric | Value |
|--------|-------|
| Total Data Size | {inventory_stats.get('total_size_bytes', 0) / (1024*1024):.1f} MB |
| Total Records | {inventory_stats.get('total_record_count', 0):,} |

### Data Types Distribution

| Data Type | Count |
|-----------|-------|
"""
        
        # Add data types
        data_types = inventory_stats.get('data_types', {})
        for data_type, count in data_types.items():
            markdown += f"| {data_type.replace('_', ' ').title()} | {count} |\n"
        
        # Add objectives and KPIs
        objectives = documentation.get('workflow_definition', {}).get('general_requirements', {}).get('objectives', [])
        kpis = documentation.get('workflow_definition', {}).get('general_requirements', {}).get('kpis', [])
        
        markdown += "\n## 🎯 Objectives & KPIs\n\n"
        
        if objectives:
            markdown += "### Objectives\n\n"
            markdown += "| Name | Target | Status | Description |\n"
            markdown += "|------|--------|--------|--------------|\n"
            for obj in objectives:
                markdown += f"| {obj.get('name', 'N/A')} | {obj.get('target', 'N/A')} | {obj.get('status', 'Pending')} | {obj.get('description', 'N/A')} |\n"
            markdown += "\n"
        
        if kpis:
            markdown += "### Key Performance Indicators\n\n"
            markdown += "| Name | Target | Current Value | Status |\n"
            markdown += "|------|--------|---------------|--------|\n"
            for kpi in kpis:
                current_val = f"{kpi.get('current_value', 'N/A')} {kpi.get('measurement_unit', '')}"
                markdown += f"| {kpi.get('name', 'N/A')} | {kpi.get('target', 'N/A')} | {current_val} | {kpi.get('status', 'Pending')} |\n"
            markdown += "\n"
        
        if not objectives and not kpis:
            markdown += "*No objectives or KPIs defined yet.*\n\n"
        
        markdown += f"""
## 📋 Technical Details

For complete technical documentation, see the generated JSON file or HTML report.

---
*Generated by Workflow Management System - {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""
        
        return markdown
    
    def generate_summary_json(self, output_path: Optional[str] = None) -> str:
        """
        Generate a JSON summary file.
        
        Args:
            output_path: Path for the JSON file
            
        Returns:
            Path to generated JSON file
        """
        if not output_path:
            output_path = os.path.join(
                self.workflow_manager.output_folder,
                "reports",
                f"workflow_summary_{self.workflow_id[:8]}.json"
            )
        
        summary = self.workflow_manager.get_workflow_summary()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Generated JSON summary: {output_path}")
        return output_path
    
    def generate_all_reports(self, base_path: Optional[str] = None) -> Dict[str, str]:
        """
        Generate all types of reports.
        
        Args:
            base_path: Base path for all reports
            
        Returns:
            Dictionary with paths to all generated reports
        """
        if not base_path:
            base_path = os.path.join(self.workflow_manager.output_folder, "reports")
        
        reports = {}
        
        try:
            # Generate HTML report
            html_path = os.path.join(base_path, f"workflow_report_{self.workflow_id[:8]}.html")
            reports['html'] = self.generate_html_report(html_path)
            
            # Generate Markdown report
            md_path = os.path.join(base_path, f"workflow_report_{self.workflow_id[:8]}.md")
            reports['markdown'] = self.generate_markdown_report(md_path)
            
            # Generate JSON summary
            json_path = os.path.join(base_path, f"workflow_summary_{self.workflow_id[:8]}.json")
            reports['json_summary'] = self.generate_summary_json(json_path)
            
            # Generate full documentation export
            full_export_path = os.path.join(base_path, f"workflow_full_export_{self.workflow_id[:8]}.json")
            reports['full_export'] = self.workflow_manager.export_workflow_package(full_export_path)
            
            logger.info(f"Generated all reports in {base_path}")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
        
        return reports
