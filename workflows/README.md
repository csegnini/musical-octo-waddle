# 🚀 Workflow Management Package

A comprehensive workflow orchestration system for multi-agent environments. This package provides enterprise-grade workflow management, state tracking, agent coordination, data inventory management, and automated documentation generation.

## 🌟 Features

### Core Capabilities
- **🔄 Workflow Orchestration**: Complete lifecycle management from initialization to completion
- **🤖 Agent Coordination**: Multi-agent task distribution and work order management
- **📊 State Management**: Real-time workflow state tracking with execution steps
- **📁 Data Inventory**: Comprehensive data reference tracking with provenance
- **📄 Documentation Generation**: Automated HTML, Markdown, and JSON report generation
- **🔒 Thread-Safe Operations**: Concurrent access support with proper locking

### Advanced Features
- **🎯 Objectives & KPIs**: Goal tracking with measurable performance indicators
- **📈 Progress Monitoring**: Real-time progress tracking across workflow stages
- **🔍 Data Lineage**: Complete data provenance and transformation history
- **⚡ Auto-Assignment**: Intelligent agent-to-task matching based on capabilities
- **📱 Multiple Output Formats**: HTML, Markdown, JSON reports and exports
- **🛡️ Error Handling**: Robust error management with detailed logging

## 📦 Package Structure

```
src/workflows/
├── __init__.py                 # Package initialization and exports
├── workflow_manager.py         # Main orchestration class (500+ lines)
├── workflow_state.py          # State management and tracking (400+ lines)
├── data_inventory.py          # Data reference and inventory management (500+ lines)
├── agent_coordinator.py       # Agent coordination and work orders (600+ lines)
├── workflow_updater.py        # Thread-safe update interface (400+ lines)
├── documentation_engine.py    # Report generation engine (400+ lines)
├── examples.py                # Usage examples and demonstrations
└── README.md                  # This documentation
```

## 🚀 Quick Start

### Basic Workflow Creation

```python
from src.workflows import WorkflowManager, DocumentationEngine

# Create a new workflow
workflow_manager = WorkflowManager(
    user_query="Analyze customer data and generate insights",
    output_folder="./workflows/my_analysis"
)

# Start the workflow
workflow_manager.start_workflow()

# Add objectives and KPIs
workflow_manager.state.add_objective(
    name="Data Analysis",
    description="Complete analysis of customer data",
    target="100% completion",
    priority="high"
)

workflow_manager.state.add_kpi(
    name="Processing Speed",
    description="Data processing rate",
    target=1000,
    measurement_unit="records/minute"
)

# Generate documentation
doc_engine = DocumentationEngine(workflow_manager)
reports = doc_engine.generate_all_reports()
```

### Agent Coordination

```python
from src.workflows import WorkflowManager

workflow_manager = WorkflowManager(
    user_query="Multi-agent data processing",
    output_folder="./workflows/multi_agent"
)

# Register agents
agent_coordinator = workflow_manager.agent_coordinator

agent_coordinator.register_agent(
    agent_id="data_processor",
    agent_type="data_processing",
    capabilities=["csv_processing", "data_validation"],
    max_concurrent_tasks=3
)

# Create work orders
work_order = agent_coordinator.create_work_order(
    task_type="data_processing",
    description="Process customer data CSV",
    requirements=["csv_processing"],
    priority="high"
)

# Auto-assign work orders
assignments = agent_coordinator.auto_assign_work_orders()
```

### External Updates

```python
from src.workflows import WorkflowUpdater

# Create updater for external access
updater = WorkflowUpdater(workflow_manager)

# Update progress
updater.update_progress(
    stage="data_loading",
    progress_percentage=25
)

# Update agent status
updater.update_agent_status(
    agent_id="agent_001",
    status="busy",
    current_task="Processing data"
)

# Add data reference
updater.add_data_reference(
    file_path="data.csv",
    description="Customer dataset",
    data_type="csv"
)
```

## 📊 Core Classes

### WorkflowManager
Main orchestration class that coordinates all workflow components.

**Key Methods:**
- `start_workflow()` - Initialize workflow execution
- `complete_workflow()` - Mark workflow as completed
- `get_workflow_summary()` - Get comprehensive status summary
- `generate_full_documentation()` - Export complete workflow documentation

### WorkflowState
Manages workflow execution state and progress tracking.

**Key Features:**
- Execution step tracking
- Objective and KPI management
- Stage progress monitoring
- Status transitions (PENDING → IN_PROGRESS → COMPLETED/FAILED)

### DataInventoryManager
Handles data reference tracking and inventory management.

**Key Features:**
- Data file registration and validation
- Provenance information tracking
- Data lineage and transformation history
- Search and filtering capabilities

### AgentCoordinator
Manages multi-agent coordination and work distribution.

**Key Features:**
- Agent registration and capability tracking
- Work order creation and management
- Auto-assignment based on agent capabilities
- Performance monitoring and load balancing

### WorkflowUpdater
Thread-safe interface for external workflow updates.

**Key Features:**
- Concurrent access support
- Progress updates
- Agent status management
- Data reference additions

### DocumentationEngine
Generates comprehensive workflow documentation.

**Key Features:**
- HTML reports with interactive dashboards
- Markdown documentation
- JSON exports
- Summary generation

## 🎯 Use Cases

### 1. Multi-Agent Data Processing
```python
# Set up agents for different processing stages
workflow_manager.agent_coordinator.register_agent("data_loader", "loading", ["csv", "json"])
workflow_manager.agent_coordinator.register_agent("data_cleaner", "cleaning", ["validation", "transformation"])
workflow_manager.agent_coordinator.register_agent("analyzer", "analysis", ["statistics", "ml"])

# Create processing pipeline
for stage in ["loading", "cleaning", "analysis"]:
    workflow_manager.agent_coordinator.create_work_order(
        task_type=stage,
        description=f"Execute {stage} stage",
        requirements=[stage]
    )
```

### 2. Machine Learning Pipeline
```python
# Track ML workflow objectives
workflow_manager.state.add_objective("Model Training", "Train ML model", "95% accuracy")
workflow_manager.state.add_kpi("Model Accuracy", "Current model performance", 0.95, "accuracy")

# Register ML-specific agents
workflow_manager.agent_coordinator.register_agent("trainer", "ml", ["sklearn", "tensorflow"])
workflow_manager.agent_coordinator.register_agent("evaluator", "evaluation", ["metrics", "validation"])
```

### 3. Data Science Research
```python
# Track research objectives
workflow_manager.state.add_objective("Hypothesis Testing", "Test research hypothesis", "p < 0.05")
workflow_manager.state.add_kpi("Statistical Power", "Study statistical power", 0.8, "power")

# Add research data
workflow_manager.data_inventory.add_data_reference(
    file_path="research_data.csv",
    description="Experimental dataset",
    data_type="csv",
    metadata={"study_id": "EXP001", "participants": 100}
)
```

## 📄 Documentation Outputs

The package generates multiple documentation formats:

### HTML Reports
- **Interactive dashboards** with progress charts
- **Agent status panels** with real-time updates
- **Data inventory tables** with search capabilities
- **Execution timeline** with step-by-step breakdown

### Markdown Documentation
- **Summary reports** for quick overview
- **Progress tracking** in tabular format
- **Agent coordination** status
- **Data inventory** listings

### JSON Exports
- **Complete workflow state** for programmatic access
- **Summary statistics** for analysis
- **Full documentation** for archival

## 🔧 Configuration

### Environment Setup
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set output directories
output_base = "./workflows"
```

### Workflow Customization
```python
# Custom workflow configuration
workflow_manager = WorkflowManager(
    user_query="Custom analysis workflow",
    output_folder="./custom_workflow",
    workflow_config={
        "enable_auto_documentation": True,
        "documentation_interval": 300,  # 5 minutes
        "max_concurrent_agents": 5,
        "enable_data_validation": True
    }
)
```

## 🔍 Examples

Run the included examples to see the package in action:

```bash
python src/workflows/examples.py
```

This will demonstrate:
1. **Basic workflow creation** and management
2. **Agent coordination** with work orders
3. **External updates** using WorkflowUpdater
4. **Complete lifecycle** from start to finish

## 🛡️ Error Handling

The package includes comprehensive error handling:

```python
try:
    workflow_manager.start_workflow()
    # ... workflow operations
    workflow_manager.complete_workflow()
except WorkflowError as e:
    logger.error(f"Workflow error: {e}")
    workflow_manager.fail_workflow(str(e))
```

## 📈 Performance

- **Thread-safe operations** for concurrent access
- **Efficient data structures** for large-scale workflows
- **Optimized documentation generation** with lazy loading
- **Memory-efficient state management** with periodic cleanup

## 🔮 Integration

### With Existing Systems
```python
# Integration with existing agent frameworks
class MyAgentFramework:
    def __init__(self):
        self.workflow_updater = WorkflowUpdater(workflow_manager)
    
    def on_task_complete(self, task_id, result):
        self.workflow_updater.update_progress(
            stage=task_id,
            progress_percentage=100
        )
```

### With Flask Applications
```python
from flask import Flask, jsonify
from src.workflows import WorkflowManager

app = Flask(__name__)
workflow_manager = WorkflowManager("API workflow", "./api_workflows")

@app.route('/workflow/status')
def get_workflow_status():
    return jsonify(workflow_manager.get_workflow_summary())
```

## 🤝 Contributing

This package is designed for enterprise multi-agent systems. Key areas for extension:

1. **Custom Agent Types** - Extend agent capabilities
2. **Additional Documentation Formats** - Add new report types
3. **Advanced Scheduling** - Implement complex task scheduling
4. **Performance Metrics** - Add detailed performance tracking

## 📝 License

Part of the Multi-Agent Scientist System - Internal Use

---

**🎉 Ready to orchestrate your multi-agent workflows!**

For more examples and advanced usage, see `examples.py` and explore the generated documentation files.
