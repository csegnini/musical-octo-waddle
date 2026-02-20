"""
Workflow Management Package - Working Demo

This file demonstrates the core functionality of the workflow management package
using the correct method signatures and parameters.
"""

import os
import sys

# Add the root directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.workflows import (
        WorkflowManager,
        WorkflowUpdater,
        DocumentationEngine,
        WorkflowStatus
    )
    from src.workflows.data_inventory import DataType
    print("✅ Successfully imported workflow management package")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def simple_demo():
    """
    Simple demonstration of the workflow management package
    """
    print("\n🚀 Workflow Management Package Demo")
    print("=" * 50)
    
    try:
        # 1. Create a workflow
        print("1️⃣ Creating workflow...")
        workflow_manager = WorkflowManager(
            user_query="Demo workflow for testing package functionality",
            output_folder=r"c:\Users\csegn\Scientist\root\workflows\simple_demo"
        )
        print(f"   ✅ Workflow ID: {workflow_manager.workflow_id[:8]}...")
        print(f"   📁 Output folder: {workflow_manager.output_folder}")
        
        # 2. Update workflow status
        print("\n2️⃣ Starting workflow...")
        workflow_manager.update_workflow_status(
            WorkflowStatus.DATA_GATHERING, 
            "Starting demo workflow"
        )
        print("   ✅ Status updated to DATA_GATHERING")
        
        # 3. Add objectives and KPIs
        print("\n3️⃣ Adding objectives and KPIs...")
        workflow_manager.add_objective(
            name="Demo Completion",
            description="Complete the workflow demo successfully",
            target="100% completion"
        )
        
        workflow_manager.add_kpi(
            name="Demo Score",
            description="Overall demo quality score",
            target="95",
            measurement_unit="%"
        )
        print("   ✅ Added 1 objective and 1 KPI")
        
        # 4. Register an agent
        print("\n4️⃣ Registering agents...")
        workflow_manager.register_agent(
            agent_id="demo_agent_001",
            name="Demo Processing Agent",
            capabilities=["data_processing", "demo_tasks"]
        )
        print("   ✅ Registered 1 agent")
        
        # 5. Add a data file
        print("\n5️⃣ Adding data files...")
        workflow_manager.add_data_file(
            data_type=DataType.UPLOADED,
            file_path=r"c:\Users\csegn\Scientist\root\data_bases\test.csv",
            name="Demo Data",
            description="Sample data for demo"
        )
        print("   ✅ Added 1 data file")
        
        # 6. Add an execution step
        print("\n6️⃣ Adding execution steps...")
        workflow_manager.add_execution_step(
            step_id="demo_step_001",
            macro_process_name="demo_processing",
            process_description="Process demo data",
            responsible_agent_or_tool="demo_agent_001"
        )
        print("   ✅ Added 1 execution step")
        
        # 7. Create a work order
        print("\n7️⃣ Creating work orders...")
        work_order = workflow_manager.create_work_order(
            instructions="Process the demo data and generate insights",
            macro_process="demo_processing",
            priority=1
        )
        print("   ✅ Created 1 work order")
        
        # 8. Generate workflow summary
        print("\n8️⃣ Generating workflow summary...")
        summary = workflow_manager.get_workflow_summary()
        print(f"   📊 Status: {summary.get('current_status', 'Unknown')}")
        print(f"   📊 User Query: {summary.get('user_query', 'N/A')}")
        print(f"   📊 Data Files: {summary.get('data_inventory_summary', {}).get('total_data_references', 0)}")
        print(f"   📊 Agents: {summary.get('agent_coordination_summary', {}).get('total_agents', 0)}")
        
        # 9. Generate documentation
        print("\n9️⃣ Generating documentation...")
        doc_engine = DocumentationEngine(workflow_manager)
        
        # Create the reports directory
        reports_dir = os.path.join(workflow_manager.output_folder, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate reports
        html_report = doc_engine.generate_html_report()
        md_report = doc_engine.generate_markdown_report()
        json_summary = doc_engine.generate_summary_json()
        
        print(f"   📄 HTML Report: {os.path.basename(html_report)}")
        print(f"   📄 Markdown Report: {os.path.basename(md_report)}")
        print(f"   📄 JSON Summary: {os.path.basename(json_summary)}")
        
        # 10. Demonstrate WorkflowUpdater
        print("\n🔟 Testing WorkflowUpdater...")
        updater = WorkflowUpdater(workflow_manager)
        
        # Add another objective
        updater.add_objective(
            name="Documentation Quality",
            description="Ensure high-quality documentation",
            target="Excellent"
        )
        print("   ✅ Added additional objective")
        
        # 11. Final status
        print("\n✅ Demo completed successfully!")
        workflow_manager.update_workflow_status(
            WorkflowStatus.COMPLETED,
            "Demo workflow completed successfully"
        )
        
        # Show final summary
        final_summary = workflow_manager.get_workflow_summary()
        print(f"\n📊 Final Summary:")
        print(f"   Status: {final_summary.get('current_status')}")
        print(f"   Objectives: {len(final_summary.get('execution_summary', {}).get('objectives', []))}")
        print(f"   KPIs: {len(final_summary.get('execution_summary', {}).get('kpis', []))}")
        print(f"   Execution Steps: {final_summary.get('execution_summary', {}).get('total_steps', 0)}")
        
        print(f"\n🎉 All features demonstrated successfully!")
        print(f"📁 Check output folder: {workflow_manager.output_folder}")
        
        return workflow_manager
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return None


def integration_example():
    """
    Show how to integrate the package into existing systems
    """
    print("\n🔗 Integration Example")
    print("-" * 30)
    
    print("To integrate this package into your multi-agent system:")
    print()
    print("```python")
    print("from src.workflows import WorkflowManager, WorkflowUpdater")
    print()
    print("# Create workflow when user submits a query")
    print("workflow = WorkflowManager(")
    print("    user_query=user_input,")
    print("    output_folder=f'./workflows/{session_id}'")
    print(")")
    print()
    print("# Let agents update the workflow")
    print("updater = WorkflowUpdater(workflow)")
    print("updater.update_progress(stage='analysis', progress_percentage=50)")
    print("updater.add_data_reference(file_path='results.csv', description='Analysis results')")
    print()
    print("# Generate reports at any time")
    print("from src.workflows import DocumentationEngine")
    print("doc_engine = DocumentationEngine(workflow)")
    print("reports = doc_engine.generate_all_reports()")
    print("```")
    print()


if __name__ == "__main__":
    print("🔬 Workflow Management Package")
    print("Testing core functionality...")
    
    # Run the demo
    result = simple_demo()
    
    if result:
        # Show integration example
        integration_example()
        
        print("\n🎯 Next Steps:")
        print("1. Open the generated HTML report in your browser")
        print("2. Review the workflow documentation structure") 
        print("3. Integrate WorkflowManager into your multi-agent system")
        print("4. Use WorkflowUpdater for real-time updates")
        print("5. Generate professional reports automatically")
    else:
        print("\n❌ Demo failed - check error messages above")
