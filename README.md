---
title: AI Research Agent
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8501
pinned: false
---

# AI Research Agent

An agentic AI system built with Google Gemini, using a planner -> executor -> synthesizer
architecture. Automatically decides whether a goal needs one step or several, calls tools
(web search via Tavily, calculator) as needed, and remembers conversation context.

## Tech stack
- Google Gemini (gemini-2.5-flash) for planning, tool-calling, and synthesis
- Tavily API for real web search
- Streamlit for the UI
- Retry/backoff logic for rate-limit handling
