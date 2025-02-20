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
        api_resource = build_resource_service(credentials=credentials)
        toolkit = GmailToolkit(api_resource=api_resource)
        tools = toolkit.get_tools()
        logger.info(f"Successfully initialized {len(tools)} Gmail tools")
        
        return {
            "api_resource": api_resource,
            "toolkit": toolkit,
            "tools": tools,
        }
    except Exception as e:
        logger.error(f"Error initializing resources: {str(e)}", exc_info=True)
        # We must update at least one of the state fields
        return {
            "errors": [{"step": "initialize_resources", "error_message": str(e)}],
            "tools": [],  # Return empty tools to ensure state is properly updated
            "api_resource": None,
            "toolkit": None
        }


async def retrieve_emails(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Retrieve recent emails from Gmail that need replies."""
    configuration = Configuration.from_runnable_config(config)
    logger.info(f"Retrieving emails from last {configuration.days_back} days")
    
    try:
        # Based on error message, the actual tool names have underscores instead of hyphens
        search_emails_tool = next((tool for tool in state.tools if tool.name == "search_gmail"), None)
        get_thread_tool = next((tool for tool in state.tools if tool.name == "get_gmail_thread"), None)
        get_message_tool = next((tool for tool in state.tools if tool.name == "get_gmail_message"), None)
        
        logger.info(f"Available tools: {[tool.name for tool in state.tools]}")
        
        if not all([search_emails_tool, get_thread_tool, get_message_tool]):
            missing_tools = []
            if not search_emails_tool: missing_tools.append("search_gmail")
            if not get_thread_tool: missing_tools.append("get_gmail_thread")
            if not get_message_tool: missing_tools.append("get_gmail_message")
            
            err_msg = f"Required Gmail tools not found: {', '.join(missing_tools)}"
            logger.error(err_msg)
            raise ValueError(err_msg)
            
        # Calculate date for query
        date_cutoff = (datetime.now() - timedelta(days=configuration.days_back)).strftime('%Y/%m/%d')
        
        # Search for recent emails that don't have replies
        query = f"{configuration.search_query} after:{date_cutoff}"
        if configuration.ignore_newsletters:
            query += " -category:promotions -category:social"
        
        logger.info(f"Searching emails with query: {query}")
        search_results = await search_emails_tool.ainvoke(query)
        logger.info(f"Search results type: {type(search_results)}")
        # logger.info(f"Search results content: {search_results[:5] if isinstance(search_results, list) else search_results}")
        
        # Handle both dictionary with 'messages' key and direct list of messages
        messages = []
        if isinstance(search_results, dict) and 'messages' in search_results:
            messages = search_results['messages']
        elif isinstance(search_results, list):
            messages = search_results
        
        if not messages:
            logger.info("No messages found matching search criteria")
            return {"emails": [], "emails_retrieved": True}
        
        logger.info(f"Found {len(messages)} messages in search results")
            
        # Limit number of emails processed
        thread_ids = [msg['threadId'] for msg in messages[:configuration.max_emails]]
        
        # Let's examine the get_thread_tool to understand its parameters
        tool_schema = getattr(get_thread_tool, 'args_schema', None)
        logger.info(f"Thread tool schema: {tool_schema}")
        
        emails_to_process = []
        for thread_id in thread_ids:
            try:
                # Try with thread_id parameter (Pydantic error suggests this is the correct name)
                logger.info(f"Getting thread with ID: {thread_id}")
                thread = await get_thread_tool.ainvoke({"thread_id": thread_id})
                
                # Get the latest message in thread that's not from the user
                latest_message_id = None
                for message in reversed(thread.get('messages', [])):
                    headers = {h['name']: h['value'] for h in message.get('payload', {}).get('headers', [])}
                    if 'me' not in headers.get('From', '').lower():
                        latest_message_id = message['id']
                        break
                        
                if latest_message_id:
                    try:
                        # Try with message_id parameter
                        logger.info(f"Getting message with ID: {latest_message_id}")
                        message_data = await get_message_tool.ainvoke({"message_id": latest_message_id})
                        
                        # Get plain text body
                        body = _extract_email_body(message_data)
                        
                        # Check if sender is in priority list
                        sender = message_data.get('sender', '')
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
                    except Exception as e:
                        logger.error(f"Error getting message {latest_message_id}: {str(e)}")
            except Exception as e:
                logger.error(f"Error getting thread {thread_id}: {str(e)}")
                
        logger.info(f"Retrieved {len(emails_to_process)} emails to process")
        return {
            "emails": emails_to_process,
            "emails_retrieved": True,
            "total_emails_processed": len(emails_to_process)
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
    
    # Handle different possible message structures
    try:
        # Check if the message has a 'body' field directly (in case of different API response format)
        if 'body' in message_data:
            return message_data.get('body', '')
            
        if 'snippet' in message_data:
            # Use snippet as fallback
            body = message_data.get('snippet', '')
                 
    except Exception as e:
        logger.error(f"Error extracting email body: {str(e)}")
        # If we encountered an error, return snippet or empty string
        if 'snippet' in message_data:
            return message_data['snippet']
            
    if not body and 'snippet' in message_data:
        return message_data['snippet']
            
    return body


def _decode_body(encoded_data):
    """Decode base64 email body"""
    if not encoded_data:
        return ""
        
    try:
        # Gmail API encodes with URL-safe base64
        # Add padding if needed
        padded_data = encoded_data + '=' * (4 - len(encoded_data) % 4) if len(encoded_data) % 4 != 0 else encoded_data
        text = base64.urlsafe_b64decode(padded_data).decode('utf-8', errors='replace')
        logger.info(f"Successfully decoded body: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error decoding body: {str(e)}")
        # If decoding fails, return the original data
        if isinstance(encoded_data, str):
            return encoded_data
        return ""


async def prioritize_emails(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Assign priority scores to emails and suggest response options."""
    configuration = Configuration.from_runnable_config(config)
    
    if not state.emails:
        return {"emails_prioritized": True}
    
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
        
        prioritize_prompt = PromptTemplate(
            input_variables=["email_info", "response_style"],
            template="""
            Analyze this email and assign a priority score from 1-10 (10 being highest priority).
            Consider factors like urgency, sender importance, and if a decision/action is required.
            Also determine if this email requires the user to make a decision before responding.

            EMAIL INFORMATION:
            {email_info}
            
            PREFERRED RESPONSE STYLE: {response_style}
            
            Provide your analysis in a JSON object with the following structure:
            {{
                "priority_score": <score>,
                "requires_decision": <true/false>,
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
            {email.body[:1000]}...
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
                    # Fallback default values
                    result = {
                        "priority_score": 5,
                        "requires_decision": True,
                        "reasoning": "Error parsing model output",
                        "proposed_response_options": [
                            "Acknowledge receipt", 
                            "Ask for more information",
                            "Forward to relevant person"
                        ]
                    }
                
                # Update email with analysis results
                email.priority_score = result.get("priority_score", 5)
                email.requires_decision = result.get("requires_decision", True)
                email.reasoning = result.get("reasoning", "")
                
                # Fix empty response options issue
                proposed_options = result.get("proposed_response_options", [])
                if not proposed_options or len(proposed_options) == 0:
                    proposed_options = [
                        "Acknowledge receipt", 
                        "No response needed",
                        "Forward if relevant"
                    ]
                
                email.proposed_response_options = proposed_options
                
                # Check if auto-reply is possible based on configuration
                if (configuration.enable_auto_replies and
                        email.priority_score <= configuration.auto_reply_threshold and
                        not email.requires_decision):
                    email.requires_decision = False
                
            except Exception as e:
                logger.error(f"Error prioritizing email: {str(e)}", exc_info=True)
                # Set default values on error
                email.priority_score = 5
                email.requires_decision = True
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


async def process_user_decision(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process user decision for the current email.
    
    This function will check if a decision is required and return appropriate
    next steps based on whether a decision has been made.
    """
    current_email = state.get_current_email()
    result = {}
    
    if not current_email:
        logger.info("No current email to process, ending workflow")
        result["next"] = "end"
        return result
    
    # If this email doesn't require a decision or has already been decided
    if not current_email.requires_decision or current_email.selected_option:
        logger.info(f"Email doesn't require decision or already has selection: {current_email.selected_option}")
        result["next"] = "generate_response"
        return result
    
    # Check if we have a user decision for this email
    if current_email.message_id in state.user_decisions:
        selected_option = state.user_decisions[current_email.message_id]
        logger.info(f"Found user decision for email {current_email.message_id}: {selected_option}")
        
        # Update the email with the selected option
        current_email.selected_option = selected_option
        updated_emails = state.emails.copy()
        updated_emails[state.current_email_index] = current_email
        
        # Continue to response generation
        result["emails"] = updated_emails
        result["next"] = "generate_response"
        return result
    
    # We need a human decision - return a special "human_feedback_needed" state
    # This will signal to the main application that it should pause execution
    # and wait for user input before continuing
    logger.info(f"Awaiting user decision for email: {current_email.subject}")
    
    # Include email details so the UI can display them to the user
    result["human_feedback_needed"] = {
        "message_id": current_email.message_id,
        "subject": current_email.subject,
        "sender": current_email.sender,
        "priority_score": current_email.priority_score,
        "reasoning": current_email.reasoning,
        "response_options": current_email.proposed_response_options
    }
    
    # CRITICAL: This must be explicitly set to "await_user_input" for proper detection
    # in the main application
    result["next"] = "await_user_input"
    logger.info(f"Requesting user input for email: {current_email.subject}")
    return result


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
            "emails": updated_emails
        }
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        return {
            "errors": [{"step": "generate_response", "error_message": str(e)}],
            "emails": state.emails  # Return current emails to maintain state
        }


async def create_draft(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Create a draft response in Gmail."""
    current_email = state.get_current_email()
    if not current_email or not current_email.generated_response:
        return {}
    
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
        current_email.draft_id = result.get('id')
        updated_emails = state.emails.copy()
        updated_emails[state.current_email_index] = current_email
        
        # Calculate new draft total
        total_drafts = state.total_drafts_created + 1
        logger.info(f"Draft created successfully. Total drafts: {total_drafts}")
        
        return {
            "emails": updated_emails,
            "total_drafts_created": total_drafts
        }
    except Exception as e:
        logger.error(f"Error creating draft: {str(e)}", exc_info=True)
        return {
            "errors": [{"step": "create_draft", "error_message": str(e)}],
            "emails": state.emails,  # Return current emails to ensure state is properly updated
        }


def router(state: State) -> Literal["process_next_email", "end"]:
    """Route to next email or end processing."""
    if state.current_email_index >= len(state.emails) - 1:
        return "end"
    return "process_next_email"


async def process_next_email(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Move to the next email in the queue."""
    return {
        "current_email_index": state.current_email_index + 1
    }
# Define the graph
workflow = StateGraph(State, config_schema=Configuration)

# Add nodes
workflow.add_node("initialize_resources", initialize_resources)
workflow.add_node("retrieve_emails", retrieve_emails)
workflow.add_node("prioritize_emails", prioritize_emails)
workflow.add_node("process_user_decision", process_user_decision)
workflow.add_node("generate_response", generate_response)
workflow.add_node("create_draft", create_draft)
workflow.add_node("process_next_email", process_next_email)

# Add edges
workflow.add_edge("__start__", "initialize_resources")
workflow.add_edge("initialize_resources", "retrieve_emails")
workflow.add_edge("retrieve_emails", "prioritize_emails")
workflow.add_edge("prioritize_emails", "process_user_decision")

# Handle all process_user_decision routing in a single conditional edges block
# Use a custom routing function that properly handles await_user_input
def process_user_decision_router(state: State) -> str:
    # First priority: check if we need user input
    if getattr(state, "next", None) == "await_user_input":
        logger.info("Routing to end for user input")
        return "__end__"
    
    # Use the specified next state or default to generate_response
    next_step = getattr(state, "next", "generate_response")
    logger.info(f"Routing to: {next_step}")
    return next_step

workflow.add_conditional_edges(
    "process_user_decision",
    process_user_decision_router,
    {
        "generate_response": "generate_response",
        "process_user_decision": "process_user_decision",  # Allow direct routing to self
        "end": "__end__",
        "__end__": "__end__"  # Explicit mapping for clarity
    }
)

workflow.add_edge("generate_response", "create_draft")

# Router after creating draft
workflow.add_conditional_edges(
    "create_draft",
    router,
    {
        "process_next_email": "process_next_email",
        "end": "__end__"
    }
)

workflow.add_edge("process_next_email", "process_user_decision")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "Gmail Assistant Graph"