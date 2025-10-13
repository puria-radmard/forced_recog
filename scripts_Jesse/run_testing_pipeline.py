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

Mock Test Data Structure:
- mock_test_data_AT_2T_pref: Assist Tag Preference - 2 Turn
- mock_test_data_AT_2T_rec: Assist Tag Recognition - 2 Turn
- mock_test_data_AT_IR: Assist Tag Injected Response
- mock_test_data_UT_2T_pref: User Tag Preference - 2 Turn
- mock_test_data_UT_2T_rec: User Tag Recognition - 2 Turn

Usage:
    python run_testing_pipeline.py                        # Run all mock tests (~15 seconds)
    python run_testing_pipeline.py --mode individual     # Run tests individually
    python run_testing_pipeline.py --mode parallel       # Run tests via parallel script (1 worker)
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
        self.run_parallel_script = self.project_root / "scripts_Jesse" / "run_all_experiments_parallel.py"
        self.experiments_dir = self.project_root / "results_and_data" / "experiments"
        
        # Discover mock test data experiments dynamically
        self.available_tests = self._discover_mock_test_data()
    
    def _discover_mock_test_data(self) -> Dict[str, Dict[str, str]]:
        """
        Discover all mock_test_data experiments in the experiments directory.
        
        Returns:
            Dictionary mapping test names to test configurations
        """
        available_tests = {}
        
        if not self.experiments_dir.exists():
            print(f"Warning: Experiments directory not found: {self.experiments_dir}")
            return available_tests
        
        # Find all directories that start with "mock_test_data"
        for item in self.experiments_dir.iterdir():
            if item.is_dir() and item.name.startswith("mock_test_data"):
                # Extract experiment type and paradigm from directory name
                # Format: mock_test_data-{EXPERIMENT_TYPE}_{PARADIGM} or mock_test_data-{EXPERIMENT_TYPE}
                # Remove the "mock_test_data-" prefix first
                name_without_prefix = item.name.replace("mock_test_data-", "")
                
                # Parse experiment type and paradigm
                if name_without_prefix == "AT_IR":
                    experiment_type = "AT_IR"
                    paradigm = "rec"  # Default to rec
                elif name_without_prefix.startswith("AT_2T_"):
                    experiment_type = "AT_2T"
                    paradigm = name_without_prefix.split("_")[-1]  # Get the last part (pref/rec)
                elif name_without_prefix.startswith("UT_2T_"):
                    experiment_type = "UT_2T"
                    paradigm = name_without_prefix.split("_")[-1]  # Get the last part (pref/rec)
                else:
                    continue  # Unknown experiment format
                
                # Map paradigm to full name
                paradigm_names = {
                    "rec": "Recognition",
                    "pref": "Preference"
                }
                
                # Map experiment type to full name
                experiment_names = {
                    "AT_2T": "Assist Tag - 2 Turn",
                    "AT_IR": "Assist Tag - Injected Response", 
                    "UT_2T": "User Tag - 2 Turn"
                }
                
                if paradigm:
                    test_name = f"{experiment_type}_{paradigm.upper()}"
                    description = f"{experiment_names.get(experiment_type, experiment_type)} {paradigm_names.get(paradigm, paradigm)} (Mock Model)"
                    
                    # Find the config file that matches this experiment type and paradigm
                    config_file = f"{experiment_type}/{paradigm}_config_mock.yaml"
                    batch_config = f"{experiment_type}/{paradigm}_config_batch_mock.yaml"
                else:
                    # No paradigm specified - skip this directory
                    print(f"Warning: No paradigm found in directory name: {item.name}")
                    continue
                
                config_path = self.configs_dir / config_file
                
                if config_path.exists():
                    available_tests[test_name] = {
                        "config": config_file,
                        "description": description,
                        "experiment_dir": item.name,
                        "batch_config": batch_config
                    }
                else:
                    print(f"Warning: Config file not found for {test_name}: {config_path}")
        
        return available_tests
    
    def list_tests(self):
        """List all available tests."""
        print("Available Tests:")
        print("=" * 50)
        if not self.available_tests:
            print("No mock test data experiments found.")
            print(f"Looking for directories starting with 'mock_test_data' in: {self.experiments_dir}")
            return
            
        for test_name, test_info in self.available_tests.items():
            print(f"{test_name:12} - {test_info['description']}")
            print(f"              Config: {test_info['config']}")
            print(f"              Experiment Dir: {test_info['experiment_dir']}")
            print(f"              Batch Config: {test_info.get('batch_config', 'N/A')}")
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
        experiment_dir = self.experiments_dir / test_info["experiment_dir"]
        
        # For mock test data, we need to find the subdirectory containing the CSV files
        if experiment_dir.name.startswith("mock_test_data"):
            # Find the subdirectory containing control.csv and treatment.csv
            subdirs = [d for d in experiment_dir.iterdir() if d.is_dir()]
            if subdirs:
                # Use the first subdirectory (should be the only one)
                experiment_dir = subdirs[0]
            else:
                raise ValueError(f"No subdirectories found in {experiment_dir}")
        
        print(f"Running test: {test_name}")
        print(f"Config: {config_path}")
        print(f"Experiment Dir: {experiment_dir}")
        print(f"Description: {test_info['description']}")
        print("-" * 50)
        
        # Record start time
        start_time = time.time()
        
        try:
            # Run the experiment script with specific experiment directory
            cmd = [
                sys.executable,
                str(self.run_experiment_script),
                "--config", str(config_path),
                "--experiment-dir", str(experiment_dir)
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
            # Results are saved based on experiment path inference
            # For mock_test_data experiments, results go to results_and_data/results/mock_test_data_{TYPE}_{PARADIGM}/
            experiment_dir_name = test_info["experiment_dir"]  # e.g., "mock_test_data_AT_2T_rec"
            
            # Extract the subdirectory within the experiment directory
            experiment_path = self.experiments_dir / experiment_dir_name
            if experiment_path.exists():
                # Find the subdirectory (e.g., "mock_control_vs_typo_S2")
                subdirs = [d for d in experiment_path.iterdir() if d.is_dir()]
                if subdirs:
                    subdir_name = subdirs[0].name  # Take the first subdirectory
                    results_dir = self.project_root / "results_and_data" / "results" / experiment_dir_name
                    results_file = results_dir / f"{subdir_name}_choice_results.csv"
                else:
                    print(f"  Status: [ERROR] No subdirectories found in {experiment_path}")
                    continue
            else:
                print(f"  Status: [ERROR] Experiment directory not found: {experiment_path}")
                continue
            
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
        # Save report to a general testing directory
        report_dir = self.project_root / "results_and_data" / "results" / "testing_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        report_path = report_dir / f"test_report_{timestamp}.txt"
        
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
    
    def run_individual_tests(self) -> List[Dict[str, any]]:
        """Run tests individually using run_experiment.py."""
        print("Running tests individually...")
        print("=" * 50)
        
        results = []
        total_start_time = time.time()
        
        for test_name, test_info in self.available_tests.items():
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
        print("\nIndividual Testing Summary:")
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
    
    def run_parallel_tests(self) -> List[Dict[str, any]]:
        """Run tests via the parallel script with 1 worker."""
        print("Running tests via parallel script (1 worker)...")
        print("=" * 50)
        
        results = []
        total_start_time = time.time()
        
        for test_name, test_info in self.available_tests.items():
            print(f"\nRunning {test_name} via parallel script...")
            
            # Record start time
            start_time = time.time()
            
            try:
                # Use batch config for parallel script
                batch_config_path = self.configs_dir / test_info["batch_config"]
                
                if not batch_config_path.exists():
                    print(f"Error: Batch config not found: {batch_config_path}")
                    results.append({
                        "test_name": test_name,
                        "success": False,
                        "duration": 0,
                        "stdout": "",
                        "stderr": f"Batch config not found: {batch_config_path}",
                        "return_code": -1
                    })
                    continue
                
                # Create a temporary config that points to the specific mock test data directory
                temp_config_path = self.project_root / "temp_parallel_config.yaml"
                
                # Read the batch config and modify experiment_dir
                import yaml
                with open(batch_config_path, "r") as f:
                    batch_config = yaml.safe_load(f)
                
                # Don't override experiment_dir - the batch config already points to the correct directory
                # The parallel script will find subdirectories within that directory
                # batch_config["experiment_dir"] is already correct from the batch config file
                
                # Write temporary config
                with open(temp_config_path, "w") as f:
                    yaml.dump(batch_config, f)
                
                # Run the parallel script with 1 worker
                cmd = [
                    sys.executable,
                    str(self.run_parallel_script),
                    "--max-workers", "1",
                    "--config", str(temp_config_path),
                    "--continue-on-error"
                ]
                
                print(f"Command: {' '.join(cmd)}")
                
                # Run with subprocess to capture output
                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout for parallel script
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Clean up temporary config
                if temp_config_path.exists():
                    temp_config_path.unlink()
                
                success = result.returncode == 0
                if success:
                    print(f"✅ {test_name} completed successfully in {duration:.1f}s")
                else:
                    print(f"❌ {test_name} failed with exit code {result.returncode}")
                
                results.append({
                    "test_name": test_name,
                    "success": success,
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                })
                
            except subprocess.TimeoutExpired:
                end_time = time.time()
                duration = end_time - start_time
                error_msg = f"Test timed out after 5 minutes"
                print(f"❌ {test_name}: {error_msg}")
                results.append({
                    "test_name": test_name,
                    "success": False,
                    "duration": duration,
                    "stdout": "",
                    "stderr": error_msg,
                    "return_code": -1
                })
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                error_msg = f"Error running parallel test: {e}"
                print(f"❌ {test_name}: {error_msg}")
                results.append({
                    "test_name": test_name,
                    "success": False,
                    "duration": duration,
                    "stdout": "",
                    "stderr": error_msg,
                    "return_code": -1
                })
            
            print("-" * 50)
        
        total_duration = time.time() - total_start_time
        
        # Print summary
        print("\nParallel Testing Summary:")
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


def main():
    """Main function for the testing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run testing pipeline for all operationalizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_testing_pipeline.py                    # Run all tests individually (default)
  python run_testing_pipeline.py --mode individual  # Run tests individually
  python run_testing_pipeline.py --mode parallel    # Run tests via parallel script (1 worker)
  python run_testing_pipeline.py --list             # List available tests
  python run_testing_pipeline.py --clean            # Clean testing directories
  python run_testing_pipeline.py --check            # Check and display test results
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["individual", "parallel"],
        default="individual",
        help="Testing mode: individual (run_experiment.py) or parallel (run_all_experiments_parallel.py with 1 worker)"
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
    
    # Run tests based on mode
    if args.mode == "individual":
        results = pipeline.run_individual_tests()
    elif args.mode == "parallel":
        results = pipeline.run_parallel_tests()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
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
