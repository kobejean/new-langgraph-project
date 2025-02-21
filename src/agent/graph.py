"""Define the Gmail assistant agent workflow.

This agent manages emails using a multi-step workflow to retrieve, prioritize,
suggest responses, and create drafts based on user decisions.
"""

from typing import Any, Dict, List, Literal, Union
from datetime import datetime, timedelta
import re
import base64
import logging
from email.mime.text import MIMEText

from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from agent.configuration import Configuration
from agent.state import State, EmailInfo

# Set up logging
logger = logging.getLogger("gmail_assistant.graph")


async def initialize_resources(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Initialize Gmail API resources and tools."""
    configuration = Configuration.from_runnable_config(config)
    logger.info(f"Initializing resources with model: {configuration.model_name}")
    
    try:
        # Set up Gmail toolkit
        logger.info("Getting Gmail credentials...")
        credentials = get_gmail_credentials(
            token_file="token.json",
            scopes=["https://mail.google.com/"],
            client_secrets_file="credentials.json",
        )
        logger.info("Building API resource...")
        toolkit = GmailToolkit(api_resource=build_resource_service(credentials=credentials))
        tools = toolkit.get_tools()
        logger.info(f"Successfully initialized {len(tools)} Gmail tools")
        
        return {
            "tools": tools,
        }
    except Exception as e:
        logger.error(f"Error initializing resources: {str(e)}", exc_info=True)
        # We must update at least one of the state fields
        return {
            "errors": [{"step": "initialize_resources", "error_message": str(e)}],
            "tools": [],  # Return empty tools to ensure state is properly updated
        }


async def retrieve_emails(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve recent emails from Gmail that need replies."""
    configuration = Configuration.from_runnable_config(config)
    logger.info(f"Retrieving emails from last {configuration.days_back} days")
    
    try:
        search_emails_tool = next((tool for tool in state.tools if tool.name == "search_gmail"), None)
        get_thread_tool = next((tool for tool in state.tools if tool.name == "get_gmail_thread"), None)
        get_message_tool = next((tool for tool in state.tools if tool.name == "get_gmail_message"), None)
        logger.info(f"Available tools: {[tool.name for tool in state.tools]}")
        assert all([search_emails_tool, get_thread_tool, get_message_tool])

        # Calculate date for query
        date_cutoff = (datetime.now() - timedelta(days=configuration.days_back)).strftime('%Y/%m/%d')
        
        # Search for recent emails that don't have replies
        query = f"{configuration.search_query} after:{date_cutoff}"
        if configuration.ignore_newsletters:
            query += " -category:promotions -category:social"
        
        logger.info(f"Searching emails with query: {query}")
        messages = await search_emails_tool.ainvoke(query)
        
        if not messages:
            logger.info("No messages found matching search criteria")
            return {"emails": [], "emails_retrieved": True}
        
        logger.info(f"Found {len(messages)} messages in search results")
            
        # Limit number of emails processed
        thread_ids = [msg['threadId'] for msg in messages[:configuration.max_emails]]
        
        emails_to_process = []
        for thread_id in thread_ids:
            # Try with thread_id parameter (Pydantic error suggests this is the correct name)
            logger.info(f"Getting thread with ID: {thread_id}")
            thread = await get_thread_tool.ainvoke({"thread_id": thread_id})
            
            # TODO: check if sender is me
            # Get the latest message in thread 
            latest_message_id = thread['messages'][-1]['id']

            if latest_message_id:
                # Try with message_id parameter
                logger.info(f"Getting message with ID: {latest_message_id}")
                message_data = await get_message_tool.ainvoke({"message_id": latest_message_id})
                
                # Get plain text body
                body = _extract_email_body(message_data)
                
                # Check if sender is in priority list
                sender = message_data.get('sender', '(Unknown)')
                is_priority = any(priority_sender in sender.lower() 
                                    for priority_sender in configuration.priority_senders)
                
                email_info = EmailInfo(
                    thread_id=thread_id,
                    message_id=latest_message_id,
                    subject=message_data.get('subject', '(No Subject)'),
                    sender=sender,
                    date=message_data.get('Date', ''),
                    body=body,
                    # Give initial priority boost to priority senders
                    priority_score=5 if not is_priority else 8
                )
                
                emails_to_process.append(email_info)
                logger.info(f"Added email to process: {email_info.subject} from {email_info.sender}")

        logger.info(f"Retrieved {len(emails_to_process)} emails to process")
        return {
            "emails": emails_to_process,
            "emails_retrieved": True,
        }
    except Exception as e:
        logger.error(f"Error retrieving emails: {str(e)}", exc_info=True)
        return {
            "errors": [{"step": "retrieve_emails", "error_message": str(e)}],
            "emails_retrieved": False,
            "emails": []  # Return empty emails to ensure state is properly updated
        }


def _extract_email_body(message_data):
    """Extract plaintext email body from message data."""
    body = ""
    if 'body' in message_data: body = message_data['body']   
    elif 'snippet' in message_data: body = message_data['snippet']
    return body.strip()

async def prioritize_emails(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Assign priority scores to emails and suggest response options."""
    configuration = Configuration.from_runnable_config(config)
    
    if not state.emails:
        return {"emails_prioritized": True}
    
    try:
        from langchain_ollama import OllamaLLM
        logger.info(f"Using OllamaLLM from langchain_ollama with model: {configuration.model_name}")
        llm = OllamaLLM(model=configuration.model_name)
        
        prioritize_prompt = PromptTemplate(
            input_variables=["email_info", "response_style"],
            template="""
            Analyze this email and assign a priority score from 1-10 (10 being highest priority).
            Consider factors like urgency, sender importance, and if a decision/action is required.
            Also determine if we should ask the user what to respond.

            EMAIL INFORMATION:
            {email_info}
            
            PREFERRED RESPONSE STYLE: {response_style}
            
            Provide your analysis in a JSON object with the following structure:
            {{
                "priority_score": <score>,
                "ask_user": <true/false>,
                "reasoning": "<brief reason for score>",
                "proposed_response_options": [
                    "<option 1>",
                    "<option 2>",
                    "<option 3>"
                ]
            }}

            ONLY RETURN THE JSON OBJECT WITH NO ADDITIONAL TEXT BEFORE OR AFTER.
            """
        )
        
        updated_emails = []
        for email in state.emails:
            email_info = f"""
            FROM: {email.sender}
            SUBJECT: {email.subject}
            DATE: {email.date}
            BODY:
            {email.body[:2000]}...
            """
            
            try:
                logger.info(f"Prioritizing email: {email.subject}")
                # Use the direct chain with a fallback to manual parsing
                result_text = await llm.ainvoke(
                    prioritize_prompt.format(
                        email_info=email_info,
                        response_style=configuration.response_style
                    )
                )
                
                # Manual JSON extraction and parsing
                try:
                    # Extract JSON from the response
                    json_str = _extract_json(result_text)
                    import json
                    result = json.loads(json_str)
                    logger.info(f"Successfully parsed JSON: {json_str}")
                except Exception as json_err:
                    logger.error(f"Failed to parse JSON: {str(json_err)}")
                    # Fallback default values - always require a decision on error
                    result = {
                        "priority_score": 5,
                        "ask_user": True,  # Default to requiring a decision
                        "reasoning": "Error parsing model output",
                        "proposed_response_options": ["Make a draft response"]
                    }
                
                # Update email with analysis results
                email.priority_score = result.get("priority_score", 5)
                email.ask_user = result.get("ask_user", True)
                email.reasoning = result.get("reasoning", "")
                email.proposed_response_options = result.get("proposed_response_options", [])
                
            except Exception as e:
                logger.error(f"Error prioritizing email: {str(e)}", exc_info=True)
                # Set default values on error - always require a decision
                email.priority_score = 5
                email.ask_user = True
                email.reasoning = f"Error in analysis: {str(e)}"
                email.proposed_response_options = [
                    "Acknowledge receipt", 
                    "Ask for more information",
                    "Forward to relevant person"
                ]
                
            updated_emails.append(email)
                
        # Sort by priority (highest first)
        sorted_emails = sorted(updated_emails, key=lambda x: x.priority_score or 0, reverse=True)
        
        # Reset wait counter after prioritization
        return {
            "emails": sorted_emails,
            "emails_prioritized": True,
            "total_emails_processed": len(sorted_emails)  # Set the total count here
        }
    except Exception as e:
        logger.error(f"Error in prioritize_emails: {str(e)}", exc_info=True)
        return {
            "errors": [{"step": "prioritize_emails", "error_message": str(e)}],
            "emails_prioritized": False,
            "emails": state.emails,  # Return current emails to maintain state
        }


def _extract_json(text):
    """Extract JSON object from text that might contain other content."""
    import re
    import json
    
    # Find text that looks like JSON (between curly braces, including nested structures)
    json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    json_matches = re.findall(json_pattern, text)
    
    if not json_matches:
        raise ValueError("No JSON object found in the text")
    
    # Find the longest match, which is likely the full JSON object
    longest_match = max(json_matches, key=len)
    
    # Validate it's proper JSON
    json.loads(longest_match)  # This will raise an exception if invalid
    
    return longest_match


async def human_decision_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """
    Human-in-the-loop node using LangGraph's interrupt function.
    This replaces the process_user_decision function with a proper human-in-the-loop implementation.
    """
    current_email = state.get_current_email()
    if not current_email:
        logger.info("No current email to process, ending workflow")
        return {"next": "end"}
        
    # Skip human decision if email doesn't require it or already has a decision
    if not current_email.ask_user or current_email.selected_option:
        logger.info(f"No decision needed for email: {current_email.subject}")
        return {"next": "generate_response"}
        
    # Create email details for human review
    email_details = {
        "message_id": current_email.message_id,
        "subject": current_email.subject,
        "sender": current_email.sender,
        "priority_score": current_email.priority_score,
        "reasoning": current_email.reasoning,
        "response_options": current_email.proposed_response_options,
        "body_preview": current_email.body[:500] + ("..." if len(current_email.body) > 500 else "")
    }
    
    # Use interrupt to pause the workflow and wait for human input
    logger.info(f"Interrupting workflow for human decision on email: {current_email.subject}")
    
    # This is where the magic happens - we pause execution and wait for human input
    selected_option = interrupt(email_details)
    
    logger.info(f"Received human decision: {selected_option}")
    
    # Update the email with the human's decision
    if selected_option:
        # Update the email with the selected option
        current_email.selected_option = selected_option
        updated_emails = state.emails.copy()
        updated_emails[state.current_email_index] = current_email
        
        # Save the decision in user_decisions dictionary for reference
        state.user_decisions[current_email.message_id] = selected_option
        
        return {
            "emails": updated_emails,
            "user_decisions": state.user_decisions,
            "next": "generate_response"
        }
    else:
        # If no selection was made (user skipped), move to next email
        return {
            "current_email_index": state.current_email_index + 1,
            "next": "check_more_emails"  # Add this node to check if there are more emails
        }


async def check_more_emails(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Check if there are more emails to process."""
    if state.current_email_index >= len(state.emails):
        return {"next": "end"}
    return {"next": "human_decision_node"}


async def generate_response(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate email response based on selected option."""
    configuration = Configuration.from_runnable_config(config)
    
    current_email = state.get_current_email()
    if not current_email or not current_email.selected_option:
        return {}
    
    try:
        # Use OllamaLLM from langchain_ollama if available
        try:
            from langchain_ollama import OllamaLLM
            logger.info(f"Using OllamaLLM from langchain_ollama with model: {configuration.model_name}")
            llm = OllamaLLM(model=configuration.model_name)
        except ImportError:
            # Fallback to deprecated Ollama
            logger.warning(f"langchain_ollama not found, using deprecated Ollama with model: {configuration.model_name}")
            llm = Ollama(model=configuration.model_name)
        
        response_prompt = PromptTemplate(
            input_variables=[
                "subject", "sender", "body", "selected_option", 
                "response_style", "max_length", "signature"
            ],
            template="""
            Write a professional email response based on the following information and selected response approach.
            
            ORIGINAL EMAIL:
            From: {sender}
            Subject: {subject}
            Message: {body}
            
            SELECTED RESPONSE APPROACH:
            {selected_option}
            
            STYLE: {response_style}
            
            Write a complete response that:
            1. Uses an appropriate greeting
            2. Responds according to the selected approach
            3. Maintains the requested style ({response_style})
            4. Keeps the response concise (max {max_length} characters)
            5. Ends with the signature if include_signature is True
            
            SIGNATURE:
            {signature}
            
            RESPONSE:
            """
        )
        
        logger.info(f"Generating response for email: {current_email.subject}")
        logger.info(f"Selected option: {current_email.selected_option}")
        
        # Use direct invoke to simplify the call
        response_text = await llm.ainvoke(
            response_prompt.format(
                subject=current_email.subject,
                sender=current_email.sender,
                body=current_email.body[:2000],  # Limit body length
                selected_option=current_email.selected_option,
                response_style=configuration.response_style,
                max_length=configuration.max_response_length,
                signature=configuration.signature if configuration.include_signature else ""
            )
        )
        
        # Update current email with generated response
        current_email.generated_response = response_text.strip()
        updated_emails = state.emails.copy()
        updated_emails[state.current_email_index] = current_email
        
        logger.info(f"Generated response: {response_text[:200]}...")
        
        return {
            "emails": updated_emails,
            "next": "create_draft"
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return {
            "errors": [{"step": "generate_response", "error_message": str(e)}],
            "emails": state.emails,  # Return current emails to maintain state
            "next": "create_draft"  # Continue to create draft even on error
        }


async def create_draft(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Create a draft response in Gmail."""
    current_email = state.get_current_email()
    if not current_email or not current_email.generated_response:
        return {"next": "process_next_email"}
    
    try:
        # Find create draft tool - use underscores instead of hyphens
        create_draft_tool = next(
            (tool for tool in state.tools if tool.name == "create_gmail_draft"), 
            None
        )
        
        if not create_draft_tool:
            logger.error(f"Create draft tool not found. Available tools: {[tool.name for tool in state.tools]}")
            raise ValueError("Create draft tool not found")
        
        # Parse email address from the 'From' field
        from_match = re.search(r'<([^>]+)>', current_email.sender)
        if from_match:
            to_address = from_match.group(1)
        else:
            to_address = current_email.sender
            
        # Create the draft directly with proper format according to schema
        draft_request = {
            "message": current_email.generated_response,
            "to": [to_address],
            "subject": f"Re: {current_email.subject}",
            "thread_id": current_email.thread_id
        }
            
        logger.info(f"Creating draft with request format: {draft_request.keys()}")
        result = await create_draft_tool.ainvoke(draft_request)
        
        # Update current email with draft ID
        current_email.draft_id = result[-20:]
        updated_emails = state.emails.copy()
        updated_emails[state.current_email_index] = current_email
        
        return {
            "emails": updated_emails,
            "next": "process_next_email"
        }
    except Exception as e:
        logger.error(f"Error creating draft: {str(e)}", exc_info=True)
        return {
            "errors": [{"step": "create_draft", "error_message": str(e)}],
            "emails": state.emails,  # Return current emails to ensure state is properly updated
            "next": "process_next_email"  # Continue to next email even on error
        }


async def process_next_email(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Move to the next email in the queue."""
    return {
        "current_email_index": state.current_email_index + 1,
        "next": "check_more_emails"  # Go to check if there are more emails
    }


def router(state: State) -> str:
    """Route to next step based on the 'next' field in state."""
    next_step = getattr(state, "next", "check_more_emails")
    logger.info(f"Routing to: {next_step}")
    return next_step


# Define the graph
workflow = StateGraph(State, config_schema=Configuration)

# Add nodes
workflow.add_node("initialize_resources", initialize_resources)
workflow.add_node("retrieve_emails", retrieve_emails)
workflow.add_node("prioritize_emails", prioritize_emails)
workflow.add_node("human_decision_node", human_decision_node)  # New node using interrupt
workflow.add_node("generate_response", generate_response)
workflow.add_node("create_draft", create_draft)
workflow.add_node("process_next_email", process_next_email)
workflow.add_node("check_more_emails", check_more_emails)  # New node to check for more emails

# Add edges
workflow.add_edge("__start__", "initialize_resources")
workflow.add_edge("initialize_resources", "retrieve_emails")
workflow.add_edge("retrieve_emails", "prioritize_emails")
workflow.add_edge("prioritize_emails", "human_decision_node")

# Add conditional edges based on router function
workflow.add_conditional_edges(
    "human_decision_node",
    router,
    {
        "generate_response": "generate_response",
        "check_more_emails": "check_more_emails",
        "end": "__end__"
    }
)

workflow.add_conditional_edges(
    "generate_response",
    router,
    {
        "create_draft": "create_draft",
        "process_next_email": "process_next_email",
        "end": "__end__"
    }
)

workflow.add_conditional_edges(
    "create_draft",
    router,
    {
        "process_next_email": "process_next_email",
        "end": "__end__"
    }
)

workflow.add_conditional_edges(
    "check_more_emails",
    router,
    {
        "human_decision_node": "human_decision_node",
        "end": "__end__"
    }
)

workflow.add_edge("process_next_email", "check_more_emails")

# Compile the workflow into an executable graph
import os
from langgraph.graph import END

# We'll use the environment-based approach
graph = workflow.compile()
graph.name = "Gmail Assistant Graph"
graph.name = "Gmail Assistant Graph"