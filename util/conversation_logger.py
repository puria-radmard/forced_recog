"""
Conversation Logger for Experiment Tracking

This module provides a context manager for logging entire conversations
including input, model output, and results in JSON format.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class ConversationLogger:
    """
    Context manager for logging conversation data during experiments.
    
    This logger captures the entire conversation flow including:
    - Input data and prompts
    - Model responses and outputs
    - Results and analysis
    - Metadata and timing information
    """
    
    def __init__(self, 
                 experiment_name: str,
                 output_dir: str = "results_and_data/logs",
                 enabled: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize the conversation logger.
        
        Args:
            experiment_name: Name of the experiment (used for filename)
            output_dir: Directory to save log files
            enabled: Whether logging is enabled
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.enabled = enabled
        self.log_level = log_level
        
        # Create output directory if it doesn't exist
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize log data structure
        self.log_data = {
            "experiment_name": experiment_name,
            "start_time": None,
            "end_time": None,
            "conversations": [],
            "metadata": {
                "log_level": log_level,
                "logger_version": "1.0.0"
            }
        }
        
        self.current_conversation = None
        self.conversation_count = 0
    
    def __enter__(self):
        """Enter the context manager."""
        if self.enabled:
            self.log_data["start_time"] = datetime.now().isoformat()
            print(f"[INFO] Starting conversation logging for experiment: {self.experiment_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and save log file."""
        if self.enabled:
            self.log_data["end_time"] = datetime.now().isoformat()
            self._save_log_file()
            
            if exc_type is not None:
                print(f"[ERROR] Conversation logging ended with exception: {exc_type.__name__}: {exc_val}")
            else:
                print(f"[INFO] Conversation logging completed. Logged {self.conversation_count} conversations.")
    
    def start_conversation(self, 
                          conversation_id: str,
                          input_data: Dict[str, Any],
                          conversation_text: Optional[str] = None) -> None:
        """
        Start logging a new conversation.
        
        Args:
            conversation_id: Unique identifier for this conversation
            input_data: Input data for the conversation
            conversation_text: Full conversation text as the model sees it
        """
        if not self.enabled:
            return
        
        # Format conversation by splitting on "|" and creating separate parts
        formatted_conversation = None
        if conversation_text:
            # Split by "|" and clean up each part
            parts = conversation_text.split("|")
            formatted_parts = []
            for i, part in enumerate(parts, 1):
                part = part.strip()
                if part:
                    formatted_parts.append(f"part {i}: {part}")
            formatted_conversation = formatted_parts
        
        self.current_conversation = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat(),
            "input_data": input_data,
            "conversation": formatted_conversation,
            "model_responses": [],
            "results": {},
            "metadata": {}
        }
        
        if self.log_level in ["DEBUG", "INFO"]:
            print(f"[DEBUG] Started logging conversation: {conversation_id}")
    
    def log_model_response(self,
                          model_name: str,
                          response_text: str,
                          logits: Optional[Any] = None,
                          choice_probabilities: Optional[List[float]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a model response within the current conversation.
        
        Args:
            model_name: Name of the model that generated the response
            response_text: The text response from the model
            logits: Raw logits (if available)
            choice_probabilities: Extracted choice probabilities
            metadata: Additional metadata about the response
        """
        if not self.enabled or self.current_conversation is None:
            return
        
        response_data = {
            "model_name": model_name,
            "response_text": response_text,
            "timestamp": datetime.now().isoformat(),
            "choice_probabilities": choice_probabilities,
            "metadata": metadata or {}
        }
        
        # Add logits info if available (but don't store the actual tensor)
        if logits is not None:
            response_data["has_logits"] = True
            response_data["logits_shape"] = getattr(logits, 'shape', 'unknown')
            response_data["logits_type"] = type(logits).__name__
        else:
            response_data["has_logits"] = False
        
        self.current_conversation["model_responses"].append(response_data)
        
        if self.log_level in ["DEBUG", "INFO"]:
            print(f"[DEBUG] Logged response from {model_name}: '{response_text[:50]}...'")
    
    def log_results(self,
                   results: Dict[str, Any],
                   analysis: Optional[Dict[str, Any]] = None) -> None:
        """
        Log results and analysis for the current conversation.
        
        Args:
            results: Results data (e.g., accuracy, choice probabilities)
            analysis: Additional analysis data
        """
        if not self.enabled or self.current_conversation is None:
            return
        
        self.current_conversation["results"] = results
        if analysis:
            self.current_conversation["analysis"] = analysis
        
        if self.log_level in ["DEBUG", "INFO"]:
            print(f"[DEBUG] Logged results: {list(results.keys())}")
    
    def finish_conversation(self, 
                           success: bool = True,
                           error_message: Optional[str] = None) -> None:
        """
        Finish logging the current conversation.
        
        Args:
            success: Whether the conversation completed successfully
            error_message: Error message if conversation failed
        """
        if not self.enabled or self.current_conversation is None:
            return
        
        self.current_conversation["success"] = success
        self.current_conversation["end_timestamp"] = datetime.now().isoformat()
        
        if error_message:
            self.current_conversation["error_message"] = error_message
        
        # Add conversation to log data
        self.log_data["conversations"].append(self.current_conversation)
        self.conversation_count += 1
        
        if self.log_level in ["DEBUG", "INFO"]:
            status = "SUCCESS" if success else "FAILED"
            print(f"[DEBUG] Finished conversation {self.current_conversation['conversation_id']}: {status}")
        
        # Reset current conversation
        self.current_conversation = None
    
    def log_experiment_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Log experiment-level metadata.
        
        Args:
            metadata: Metadata about the experiment
        """
        if not self.enabled:
            return
        
        self.log_data["metadata"].update(metadata)
        
        if self.log_level in ["DEBUG", "INFO"]:
            print(f"[DEBUG] Updated experiment metadata: {list(metadata.keys())}")
    
    def _save_log_file(self) -> None:
        """Save the log data to a JSON file."""
        if not self.enabled:
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
            
            print(f"[INFO] Conversation log saved to: {filepath}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save conversation log: {e}")
    
    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current log data.
        
        Returns:
            Dictionary containing log summary
        """
        return {
            "experiment_name": self.experiment_name,
            "conversation_count": self.conversation_count,
            "start_time": self.log_data["start_time"],
            "end_time": self.log_data.get("end_time"),
            "enabled": self.enabled
        }


@contextmanager
def conversation_logger(experiment_name: str,
                       output_dir: str = "results_and_data/logs",
                       enabled: bool = True,
                       log_level: str = "INFO"):
    """
    Context manager factory for ConversationLogger.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory to save log files
        enabled: Whether logging is enabled
        log_level: Logging level
        
    Yields:
        ConversationLogger instance
    """
    logger = ConversationLogger(experiment_name, output_dir, enabled, log_level)
    try:
        logger.__enter__()
        yield logger
    finally:
        logger.__exit__(None, None, None)
