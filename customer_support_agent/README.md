# README: AI-Powered Customer Support Agent

## Overview

This project implements an AI-powered customer support agent using LangChain and related tools. The agent handles customer queries related to **Technical**, **Billing**, and **General** inquiries. It uses a flexible workflow to route queries based on their category and sentiment and generates context-aware responses.

## Features

1. **Query Categorization**: Automatically categorizes customer queries into predefined categories: Technical, Billing, or General.
2. **Sentiment Analysis**: Analyzes the sentiment of a query (Positive, Neutral, or Negative).
3. **Context-Aware Responses**: Generates responses tailored to the query’s category and sentiment.
4. **Escalation Handling**: Escalates queries with negative sentiment to a human agent for further assistance.
5. **Custom Workflow**: Implements a flexible workflow using LangChain's `StateGraph` for efficient state transitions.

## Installation

### Prerequisites

- Python 3.8 or higher
- API key for **Groq** to use the LLM model.

### Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   export GROQ_API_KEY=<your-groq-api-key>
   ```
4. Run the agent:
   ```bash
   python agent.py
   ```

## Project Structure

```
.
├── agent.py              # Main script for the customer support agent
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

## How It Works

### Workflow

1. **Categorization**:
   - The system categorizes the query into `Technical`, `Billing`, or `General`.
2. **Sentiment Analysis**:
   - Sentiment is analyzed to determine if the query is `Positive`, `Neutral`, or `Negative`.
3. **Routing**:
   - Based on sentiment and category:
     - Negative sentiment: Escalates to a human agent.
     - Technical/Billing/General: Generates an AI response.

### StateGraph Nodes

- **categorize**: Categorizes the query.
- **analyze\_sentiment**: Analyzes the query sentiment.
- **handle\_technical**: Handles technical support queries.
- **handle\_billing**: Handles billing support queries.
- **handle\_general**: Handles general queries.
- **escalate**: Escalates the query to a human agent.

## Example Queries

1. **Technical Query**:

   - *"My internet is not working."*
   - Response: AI provides troubleshooting steps.

2. **Billing Query**:

   - *"I was overcharged on my last bill."*
   - Response: AI generates a billing-specific response.

3. **General Query**:

   - *"What are your support hours?"*
   - Response: AI provides general information.

4. **Negative Sentiment Query**:

   - *"Your service is terrible!"*
   - Response: Query is escalated to a human agent.

## Limitations

- Requires a valid API key for Groq.
- Relies on the accuracy of the LLM model.

##





