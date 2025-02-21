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
from agent.graph import graph
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
    # Create a config with a high recursion limit
    execution_config = {
        "recursion_limit": 1000,
        "configurable": config.get("configurable", {})
    }
    
    # Initialize (only once)
    logger.info("Starting workflow...")
    initial_state = State()
    
    # Skip initialization if specified (for debugging)
    if os.environ.get("SKIP_INIT") == "1":
        logger.info("Skipping initialization (using cached state)")
        try:
            with open("cached_state.json", "r") as f:
                state_dict = json.load(f)
                emails = []
                for email_dict in state_dict.get("emails", []):
                    emails.append(EmailInfo(**email_dict))
                current_state = State(**{k: v for k, v in state_dict.items() if k != "emails"})
                current_state.emails = emails
                logger.info(f"Loaded cached state with {len(emails)} emails")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load cached state: {e}")
            current_state = initial_state
    else:
        # Do one-time initialization
        try:
            logger.info("Initializing resources and retrieving emails...")
            # We'll do these steps manually to avoid repeating them
            init_result = await graph.ainvoke(initial_state, config=execution_config)
            
            if 'emails' not in init_result or not init_result['emails']:
                logger.warning("No emails found to process")
                return
                
            # Extract fields from initialization
            valid_state_fields = {f.name for f in fields(State)}
            state_dict = {k: v for k, v in init_result.items() if k in valid_state_fields}
            current_state = State(**state_dict)
            
            # Cache state for future runs if needed
            try:
                # Convert state to dict for serialization, handling EmailInfo objects
                state_to_save = {
                    k: v for k, v in state_dict.items() if k != "emails"
                }
                if 'emails' in state_dict:
                    state_to_save['emails'] = [asdict(email) for email in state_dict['emails']]
                
                with open("cached_state.json", "w") as f:
                    json.dump(state_to_save, f)
                    logger.info("Cached state for future runs")
            except Exception as e:
                logger.warning(f"Could not cache state: {e}")
                
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            return
    
    # Track our progress
    total_processed = 0
    total_drafts = 0
    
    # Process each email one by one
    try:
        finished = False
        while not finished and current_state.current_email_index < len(current_state.emails):
            # Start processing from current email
            current_email = current_state.get_current_email()
            if not current_email:
                logger.info("No more emails to process")
                break
                
            logger.info(f"Processing email {current_state.current_email_index + 1}/{len(current_state.emails)}: {current_email.subject}")
            
            # Decision processing
            if current_email.requires_decision and not current_email.selected_option:
                # Check if user has already made a decision
                if current_email.message_id in current_state.user_decisions:
                    # Apply user decision
                    decision = current_state.user_decisions[current_email.message_id]
                    logger.info(f"Using previous decision: {decision}")
                    current_email.selected_option = decision
                    current_state.emails[current_state.current_email_index] = current_email
                else:
                    # Create human feedback request
                    email_details = {
                        "message_id": current_email.message_id,
                        "subject": current_email.subject,
                        "sender": current_email.sender,
                        "priority_score": current_email.priority_score,
                        "reasoning": current_email.reasoning,
                        "response_options": current_email.proposed_response_options
                    }
                    
                    # Get user decision
                    logger.info(f"Requesting user decision for email: {current_email.subject}")
                    decision = await get_user_decision(email_details)
                    
                    if decision:
                        logger.info(f"User selected: {decision}")
                        # Update user decisions dictionary
                        current_state.user_decisions[current_email.message_id] = decision
                        # Update the email object
                        current_email.selected_option = decision
                        current_state.emails[current_state.current_email_index] = current_email
                    else:
                        logger.info(f"User skipped email: {current_email.subject}")
                        # Move to next email
                        current_state.current_email_index += 1
                        continue
            
            # If email requires decision but has no options, skip it
            if current_email.requires_decision and not current_email.proposed_response_options:
                logger.warning(f"Email requires decision but has no options: {current_email.subject}")
                current_state.current_email_index += 1
                continue
                
            # Process the current email (generate response and create draft)
            if current_email.selected_option and not current_email.generated_response:
                # Generate response
                current_state.next = "generate_response"
                result = await graph.ainvoke(current_state, config=execution_config)
                
                # Update state with new info
                if 'emails' in result:
                    current_state.emails = result['emails']
                if 'total_drafts_created' in result:
                    current_state.total_drafts_created = result['total_drafts_created']
                    total_drafts = result['total_drafts_created']
                    
            # Move to next email
            logger.info(f"Finished processing email: {current_email.subject}")
            current_state.current_email_index += 1
            
        logger.info("Email processing complete")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in assistant loop: {str(e)}", exc_info=True)
    
    # Print final results
    print("\n=== Processing Complete ===")
    total_drafts = current_state.total_drafts_created
    print(f"Total emails processed: {current_state.current_email_index}")
    print(f"Total drafts created: {total_drafts}")


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