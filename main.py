"""Main application for the Gmail assistant.

This module provides the entry point and orchestration for the Gmail assistant.
It runs the graph-based workflow and handles user interactions.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import os
import sys
import json
from dataclasses import fields, asdict
from pprint import pprint

from langchain_core.runnables import RunnableConfig
# Import functions directly from graph.py
from agent.graph import (
    initialize_resources, 
    retrieve_emails, 
    prioritize_emails, 
    generate_response, 
    create_draft
)
from agent.state import State, EmailInfo

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")


async def get_user_decision(email_details: Dict[str, Any]) -> Optional[str]:
    """
    Function to get user decision from console.
    In a real application, this would be a UI element or API endpoint.
    
    Args:
        email_details: Details about the email requiring a decision
        
    Returns:
        Selected option or None if skipped
    """
    print("\n" + "="*50)
    print(f"DECISION REQUIRED: Email from {email_details['sender']}")
    print(f"Subject: {email_details['subject']}")
    print(f"Priority: {email_details['priority_score']}/10")
    print(f"Analysis: {email_details['reasoning']}")
    
    # Show email preview if available
    if 'body_preview' in email_details:
        print(f"\nPreview: {email_details['body_preview']}")
        
    print("\nResponse Options:")
    
    if not email_details['response_options'] or len(email_details['response_options']) == 0:
        # Provide default options if none were suggested
        email_details['response_options'] = [
            "Acknowledge receipt", 
            "Request more information",
            "No response needed"
        ]
    
    for i, option in enumerate(email_details['response_options']):
        print(f"  {i+1}. {option}")
    
    print("  0. Skip this email")
    
    while True:
        try:
            choice = input("\nSelect an option (0-{0}): ".format(len(email_details['response_options'])))
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(email_details['response_options']):
                return email_details['response_options'][choice_num-1]
            else:
                print(f"Please enter a number between 0 and {len(email_details['response_options'])}")
        except ValueError:
            print("Please enter a valid number")


async def run_assistant(config: RunnableConfig):
    """
    Run the email assistant workflow with user interaction.
    
    Args:
        config: Configuration for the assistant
    """
    # Set up the Gmail assistant without relying on LangGraph checkpointing
    thread_id = os.environ.get("THREAD_ID", f"gmail-assistant-{os.getpid()}")
    
    logger.info(f"Starting workflow with thread ID: {thread_id}")
    
    # Skip initialization if specified (for debugging)
    if os.environ.get("SKIP_INIT") == "1":
        logger.info("Skipping initialization (using cached state)")
        try:
            with open("cached_state.json", "r") as f:
                state_dict = json.load(f)
                emails = []
                for email_dict in state_dict.get("emails", []):
                    emails.append(EmailInfo(**email_dict))
                state = State(**{k: v for k, v in state_dict.items() if k != "emails"})
                state.emails = emails
                logger.info(f"Loaded cached state with {len(emails)} emails")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load cached state: {e}")
            state = State()
    else:
        state = State()
    
    # Track progress
    total_processed = 0
    
    try:
        # Initialize resources and retrieve emails first
        logger.info("Initializing resources...")
        resources_result = await initialize_resources(state, config)
        
        # Update state with returned values
        for key, value in resources_result.items():
            setattr(state, key, value)
        
        if not state.tools:
            logger.error("Failed to initialize resources")
            return
            
        logger.info("Retrieving emails...")
        emails_result = await retrieve_emails(state, config)
        
        # Update state with returned values
        for key, value in emails_result.items():
            setattr(state, key, value)
        
        if not state.emails:
            logger.info("No emails to process")
            return
            
        logger.info("Prioritizing emails...")
        prioritize_result = await prioritize_emails(state, config)
        
        # Update state with returned values
        for key, value in prioritize_result.items():
            setattr(state, key, value)
        
        # Process emails one by one with human decisions
        while state.current_email_index < len(state.emails):
            current_email = state.get_current_email()
            if not current_email:
                break
                
            logger.info(f"Processing email {state.current_email_index + 1}/{len(state.emails)}: {current_email.subject}")
            logger.info(f"{current_email.ask_user} {current_email.selected_option}")
            
            if current_email.ask_user and not current_email.selected_option:
                # Prepare email details for human review
                email_details = {
                    "message_id": current_email.message_id,
                    "subject": current_email.subject,
                    "sender": current_email.sender,
                    "priority_score": current_email.priority_score,
                    "reasoning": current_email.reasoning,
                    "response_options": current_email.proposed_response_options,
                    "body_preview": current_email.body[:500] + ("..." if len(current_email.body) > 500 else "")
                }
                
                # Get human decision
                logger.info(f"Requesting user decision for email: {current_email.subject}")
                decision = await get_user_decision(email_details)
                
                if decision:
                    logger.info(f"User selected: {decision}")
                    # Update the email with the selected option
                    current_email.selected_option = decision
                    state.emails[state.current_email_index] = current_email
                    
                    # Generate response
                    logger.info(f"Generating response...")
                    response_result = await generate_response(state, config)
                    for key, value in response_result.items():
                        setattr(state, key, value)
                    
                    # Create draft
                    logger.info(f"Creating draft...")
                    draft_result = await create_draft(state, config)
                    for key, value in draft_result.items():
                        setattr(state, key, value)
                    
                    total_processed += 1
                else:
                    logger.info(f"User skipped email: {current_email.subject}")
            
            # Move to next email
            state.current_email_index += 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in assistant loop: {str(e)}", exc_info=True)
    
    # Print final results
    print("\n=== Processing Complete ===")


def main():
    """Main function to run the assistant."""
    # Load configuration
    config = {
        "configurable": {
            "model_name": "llama3",
            "days_back": 4,
            "max_emails": 10,
            "response_style": "professional",
            "include_signature": True,
            "signature": "Best regards,\nYour Name"
        }
    }
    
    try:
        # Run the assistant
        asyncio.run(run_assistant(config))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"\nApplication error: {str(e)}")


if __name__ == "__main__":
    main()