#!/usr/bin/env python3
"""
Testing Pipeline Runner - Mock Model Integration Tests

This script runs fast integration tests using mock models to validate the entire
experiment pipeline without making LLM calls. Tests complete in seconds and cost nothing.

Test Coverage:
- All experiment types: AT_2T, AT_IR, UT_2T
- All prompt paradigms: rec (recognition), pref (preference)
- All mock datasets: capitalization, typo, all_others

Tests validate:
- Conversation generation logic
- Model loading and initialization
- Chat formatting for all experiment types
- Forward pass and logits processing
- Result calculation and storage
- Logging system functionality
- File I/O operations
- Prompt loading from consolidated YAML files

Dataset Coverage:
- mock_control_vs_typo_S2: Tested by AT_2T_REC, UT_2T_REC
- mock_control_vs_capitalization_S2: Tested by AT_2T_PREF, AT_IR_REC
- mock_vs_all_others_control_comparison: Tested by UT_2T_PREF (large dataset, tested once)

Usage:
    python run_testing_pipeline.py                        # Run all mock tests (~15 seconds)
    python run_testing_pipeline.py --config AT_2T_REC     # Run specific test
    python run_testing_pipeline.py --list                 # List available tests
    python run_testing_pipeline.py --clean                # Clean testing directories
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import windows_pathing_fix
windows_pathing_fix.fix_pathing()

class TestingPipeline:
    """Manages the testing pipeline for all operationalizations."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.configs_dir = self.project_root / "configs" / "operationalizations"
        self.results_testing_dir = self.project_root / "results_and_data" / "results" / "testing"
        self.logs_testing_dir = self.project_root / "results_and_data" / "logs" / "testing"
        self.run_experiment_script = self.project_root / "scripts_Jesse" / "run_experiment.py"
        
        # Define available tests (using mock models for fast integration testing)
        # Tests cover:
        # - All experiment types (AT_2T, AT_IR, UT_2T)
        # - All prompt paradigms (rec, pref)
        # - All mock datasets (capitalization, typo, all_others)
        self.available_tests = {
            "AT_2T_REC": {
                "config": "AT_2T/rec_config_mock.yaml",
                "description": "Assist Tag Recognition - 2 Turn (Mock Model)",
                "experiment_dir": "mock_test_data/mock_control_vs_typo_S2"
            },
            "AT_2T_PREF": {
                "config": "AT_2T/pref_config_mock.yaml",
                "description": "Assist Tag Preference - 2 Turn (Mock Model)",
                "experiment_dir": "mock_test_data/mock_control_vs_capitalization_S2"
            },
            "AT_IR_REC": {
                "config": "AT_IR/rec_config_mock.yaml", 
                "description": "Assist Tag Recognition - Injected Response (Mock Model)",
                "experiment_dir": "mock_test_data/mock_control_vs_capitalization_S2"
            },
            "UT_2T_REC": {
                "config": "UT_2T/rec_config_mock.yaml",
                "description": "User Tag Recognition - 2 Turn (Mock Model)", 
                "experiment_dir": "mock_test_data/mock_control_vs_typo_S2"
            },
            "UT_2T_PREF": {
                "config": "UT_2T/pref_config_mock.yaml",
                "description": "User Tag Preference - 2 Turn (Mock Model)",
                "experiment_dir": "mock_test_data/mock_vs_all_others_control_comparison"
            }
        }
    
    def list_tests(self):
        """List all available tests."""
        print("Available Tests:")
        print("=" * 50)
        for test_name, test_info in self.available_tests.items():
            print(f"{test_name:8} - {test_info['description']}")
            print(f"          Config: {test_info['config']}")
            print(f"          Experiment Dir: {test_info['experiment_dir']}")
            print()
    
    def clean_testing_dirs(self):
        """Clean all testing directories."""
        print("Cleaning testing directories...")
        
        if self.results_testing_dir.exists():
            shutil.rmtree(self.results_testing_dir)
            print(f"Cleaned: {self.results_testing_dir}")
        
        if self.logs_testing_dir.exists():
            shutil.rmtree(self.logs_testing_dir)
            print(f"Cleaned: {self.logs_testing_dir}")
        
        # Recreate empty directories
        self.results_testing_dir.mkdir(parents=True, exist_ok=True)
        self.logs_testing_dir.mkdir(parents=True, exist_ok=True)
        
        print("Testing directories cleaned and recreated.")
    
    def validate_test_setup(self) -> bool:
        """Validate that all test configs exist and are properly set up."""
        print("Validating test setup...")
        
        missing_configs = []
        for test_name, test_info in self.available_tests.items():
            config_path = self.configs_dir / test_info["config"]
            if not config_path.exists():
                missing_configs.append(str(config_path))
        
        if missing_configs:
            print("ERROR: Missing test configs:")
            for config in missing_configs:
                print(f"  - {config}")
            return False
        
        if not self.run_experiment_script.exists():
            print(f"ERROR: Missing run_experiment.py script: {self.run_experiment_script}")
            return False
        
        print("Test setup validation passed.")
        return True
    
    def run_single_test(self, test_name: str) -> Dict[str, any]:
        """Run a single test and return results."""
        if test_name not in self.available_tests:
            raise ValueError(f"Unknown test: {test_name}")
        
        test_info = self.available_tests[test_name]
        config_path = self.configs_dir / test_info["config"]
        
        print(f"Running test: {test_name}")
        print(f"Config: {config_path}")
        print(f"Description: {test_info['description']}")
        print("-" * 50)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the experiment script
            cmd = [
                sys.executable,
                str(self.run_experiment_script),
                "--config", str(config_path)
            ]
            
            print(f"Command: {' '.join(cmd)}")
            print()
            
            # Run with subprocess to capture output
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout (mock tests are fast)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "test_name": test_name,
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_name": test_name,
                "success": False,
                "duration": 60,
                "stdout": "",
                "stderr": "Test timed out after 1 minute (mock tests should complete in seconds)",
                "return_code": -1
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            return {
                "test_name": test_name,
                "success": False,
                "duration": duration,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }
    
    def run_all_tests(self) -> List[Dict[str, any]]:
        """Run all available tests."""
        print("Running all tests...")
        print("=" * 50)
        
        results = []
        total_start_time = time.time()
        
        for test_name in self.available_tests.keys():
            result = self.run_single_test(test_name)
            results.append(result)
            
            # Print result summary
            status = "PASS" if result["success"] else "FAIL"
            duration = result["duration"]
            print(f"\n{test_name}: {status} ({duration:.1f}s)")
            
            if not result["success"]:
                print(f"Error: {result['stderr']}")
            
            print("-" * 50)
        
        total_duration = time.time() - total_start_time
        
        # Print summary
        print("\nTesting Pipeline Summary:")
        print("=" * 50)
        passed = sum(1 for r in results if r["success"])
        failed = len(results) - passed
        print(f"Total Tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total Duration: {total_duration:.1f}s")
        
        if failed > 0:
            print("\nFailed Tests:")
            for result in results:
                if not result["success"]:
                    print(f"  - {result['test_name']}: {result['stderr']}")
        
        return results
    
    def check_test_results(self):
        """Check and display test results."""
        print("Checking test results...")
        
        # Check results directories
        for test_name, test_info in self.available_tests.items():
            # Results are saved to results_and_data/results/mock_test_data/ based on experiment path inference
            results_dir = self.project_root / "results_and_data" / "results" / "mock_test_data"
            experiment_name = test_info["experiment_dir"].split("/")[-1]  # Get the last part (e.g., "mock_control_vs_typo_S2")
            results_file = results_dir / f"{experiment_name}_choice_results.csv"
            
            print(f"\n{test_name}:")
            print(f"  Results File: {results_file}")
            
            if results_file.exists():
                print(f"  Status: [OK] Results file exists")
                print(f"  Size: {results_file.stat().st_size} bytes")
                print(f"  Modified: {results_file.stat().st_mtime}")
            else:
                print(f"  Status: [MISSING] Results file not found")
    
    def save_test_report(self, results: List[Dict[str, any]]):
        """Save test results to a report file."""
        # Save report to the mock_test_data results directory
        report_dir = self.project_root / "results_and_data" / "results" / "mock_test_data"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "test_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Testing Pipeline Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            passed = sum(1 for r in results if r["success"])
            failed = len(results) - passed
            
            f.write(f"Summary:\n")
            f.write(f"  Total Tests: {len(results)}\n")
            f.write(f"  Passed: {passed}\n")
            f.write(f"  Failed: {failed}\n\n")
            
            f.write("Detailed Results:\n")
            f.write("-" * 30 + "\n")
            
            for result in results:
                status = "PASS" if result["success"] else "FAIL"
                f.write(f"\n{result['test_name']}: {status}\n")
                f.write(f"  Duration: {result['duration']:.1f}s\n")
                f.write(f"  Return Code: {result['return_code']}\n")
                
                if result['stdout']:
                    f.write(f"  Output:\n{result['stdout']}\n")
                
                if result['stderr']:
                    f.write(f"  Error:\n{result['stderr']}\n")
        
        print(f"Test report saved to: {report_path}")


def main():
    """Main function for the testing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run testing pipeline for all operationalizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_testing_pipeline.py                    # Run all tests
  python run_testing_pipeline.py --config AT_2T     # Run specific test
  python run_testing_pipeline.py --list             # List available tests
  python run_testing_pipeline.py --clean            # Clean testing directories
        """
    )
    
    parser.add_argument(
        "--config",
        choices=["AT_2T_REC", "AT_2T_PREF", "AT_IR_REC", "UT_2T_REC", "UT_2T_PREF"],
        help="Run specific test configuration"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available tests"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true", 
        help="Clean all testing directories"
    )
    
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check and display test results"
    )
    
    args = parser.parse_args()
    
    pipeline = TestingPipeline()
    
    if args.list:
        pipeline.list_tests()
        return
    
    if args.clean:
        pipeline.clean_testing_dirs()
        return
    
    if args.check:
        pipeline.check_test_results()
        return
    
    # Validate setup before running tests
    if not pipeline.validate_test_setup():
        print("Test setup validation failed. Please fix the issues above.")
        sys.exit(1)
    
    # Run tests
    if args.config:
        # Run single test
        result = pipeline.run_single_test(args.config)
        results = [result]
    else:
        # Run all tests
        results = pipeline.run_all_tests()
    
    # Save test report
    pipeline.save_test_report(results)
    
    # Check results
    pipeline.check_test_results()
    
    # Exit with appropriate code
    failed_tests = [r for r in results if not r["success"]]
    if failed_tests:
        print(f"\nTesting pipeline completed with {len(failed_tests)} failures.")
        sys.exit(1)
    else:
        print("\nAll tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
