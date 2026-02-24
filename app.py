"""
app.py — Conversational Multifamily Investment Screener

A Streamlit chat interface powered by Claude that walks users through
analyzing multifamily rental data. Upload a CSV, chat naturally, and
Claude handles the analysis behind the scenes.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
import base64
import anthropic

from tools import TOOL_DEFINITIONS, execute_tool

# ─── Page config ─────────────────────────────────────────────

st.set_page_config(
    page_title="Multifamily Investment Screener",
    page_icon="🏢",
    layout="centered",
)

# ─── Custom styling ──────────────────────────────────────────

st.markdown("""
<style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .block-container { padding-top: 2rem; }
    h1 { font-size: 1.6rem !important; }
    .stChatMessage { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────

st.title("🏢 Multifamily Investment Screener")
st.caption("Upload your rental data and chat to find undervalued properties. Powered by Claude.")

# ─── Sidebar: API key + file upload ──────────────────────────

with st.sidebar:
    st.header("Setup")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Get one at console.anthropic.com. Your key stays in your browser session.",
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Yardi Matrix export or any CSV with property-level rent data.",
    )

    if uploaded_file:
        if "df" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
            st.session_state.df = pd.read_csv(uploaded_file, low_memory=False)
            st.session_state.filename = uploaded_file.name
            st.success(f"Loaded **{uploaded_file.name}** — {len(st.session_state.df):,} rows, {len(st.session_state.df.columns)} columns")

    st.divider()
    st.markdown("""
    **How to use:**
    1. Paste your Anthropic API key
    2. Upload a CSV with property data
    3. Ask questions like:
       - *"What submarkets are available?"*
       - *"Analyze Midtown West with landmarks near Georgia Tech"*
       - *"Which properties are the best deals?"*
    """)

# ─── System prompt ───────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert real estate investment analyst assistant. You help users analyze multifamily rental property data to find undervalued acquisition targets.

You have access to tools that can inspect uploaded data, list submarkets, describe available features, and run a full LASSO regression analysis.

Your workflow:
1. When the user uploads data, inspect it first to understand the structure.
2. Help them pick a submarket to analyze.
3. Suggest relevant features and landmarks based on the submarket's location.
4. Run the analysis and explain results in clear, non-technical language.
5. Highlight the most promising acquisition targets and explain WHY they look undervalued.

When explaining model results:
- Translate R² into plain English (e.g. "the model explains 73% of rent variation")
- Explain variable importance intuitively (e.g. "having a garden is associated with $0.17/sqft lower rent — likely because garden-style properties are older and less dense")
- Frame acquisition targets as investment opportunities with specific dollar amounts when possible
- Be specific about property names, addresses, and numbers

Keep responses conversational but substantive. You're talking to someone who understands real estate but may not know statistics. When using tools, make multiple calls if needed — don't ask the user to do things you can figure out from the data.

IMPORTANT: When you receive results from run_analysis that include a map_image_base64 field, tell the user you've generated a map and it will be displayed below. Do NOT try to render the image yourself."""

# ─── Chat state ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "map_images" not in st.session_state:
    st.session_state.map_images = []

# ─── Display chat history ────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    # Check if there's a map to show after this assistant message
    for img in st.session_state.map_images:
        if img["after_index"] == i:
            st.image(base64.b64decode(img["b64"]), caption="Submarket Map", use_container_width=True)

# ─── Chat input ──────────────────────────────────────────────

if prompt := st.chat_input("Ask about your data..."):
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
        st.stop()

    if "df" not in st.session_state:
        st.error("Please upload a CSV file first.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for Claude API (only user/assistant, no tool results in display)
    api_messages = []
    for msg in st.session_state.messages:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # Call Claude with tools
    client = anthropic.Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Agentic loop: keep calling Claude until it gives a final text response
            current_messages = list(api_messages)
            max_iterations = 10
            final_text = ""
            map_b64 = None

            for _ in range(max_iterations):
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,
                    messages=current_messages,
                )

                # Process response content blocks
                tool_results = []
                text_parts = []

                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_use":
                        # Execute the tool
                        tool_name = block.name
                        tool_input = block.input

                        with st.status(f"Running {tool_name}...", expanded=False):
                            result = execute_tool(tool_name, tool_input, st.session_state.df)

                            # Extract map if present
                            if isinstance(result, dict) and "map_image_base64" in result:
                                map_b64 = result.pop("map_image_base64")

                            st.json(result if isinstance(result, dict) else {"result": str(result)})

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str)[:50000],  # truncate if huge
                        })

                if response.stop_reason == "tool_use":
                    # Claude wants to use tools — add assistant message + tool results, loop again
                    current_messages.append({"role": "assistant", "content": response.content})
                    current_messages.append({"role": "user", "content": tool_results})
                else:
                    # Final response
                    final_text = "\n".join(text_parts)
                    break

            st.markdown(final_text)

            # Show map if generated
            if map_b64:
                st.image(base64.b64decode(map_b64), caption="Submarket Map", use_container_width=True)
                st.session_state.map_images.append({
                    "after_index": len(st.session_state.messages),
                    "b64": map_b64,
                })

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": final_text})
