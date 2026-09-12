# ============================================================
# AGENTIC AI CHATBOT - STREAMLIT FRONTEND
# ============================================================
#
# Features:
#
# - OpenRouter
# - LangGraph
# - SQLite memory
# - PDF RAG
# - FAISS
# - Tavily
# - Calculator
# - Stock
# - Weather
# - LangSmith
# - ChatGPT-style interface
# - PDF upload inside chat input
# - HUMAN-IN-THE-LOOP
# - Purchase approval UI
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import uuid

import streamlit as st

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langgraph.types import Command

from backend import (
    chatbot,
    get_all_threads,
    get_thread_history,
    get_pending_interrupt,
    resume_chat,
    ingest_rag_document,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="Agentic AI Chatbot",

    page_icon="🤖",

    layout="wide",

    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #0e1117;
    }

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 7rem;
        max-width: 1200px;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #171923;
        border-right: 1px solid #292c38;
    }

    .sidebar-title {
        font-size: 26px;
        font-weight: 700;
        color: white;
        margin-bottom: 4px;
    }

    .sidebar-subtitle {
        color: #9ca3af;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .sidebar-section {
        color: #9ca3af;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 28px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .app-header {
        text-align: center;
        padding: 8px 0 20px 0;
    }

    .app-title {
        font-size: 30px;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }

    .app-subtitle {
        font-size: 14px;
        color: #9ca3af;
    }


    /* ========================================================
       WELCOME
       ======================================================== */

    .welcome-container {
        text-align: center;
        padding-top: 70px;
        padding-bottom: 70px;
    }

    .welcome-icon {
        font-size: 58px;
        margin-bottom: 15px;
    }

    .welcome-title {
        font-size: 32px;
        font-weight: 700;
        color: white;
        margin-bottom: 10px;
    }

    .welcome-description {
        color: #9ca3af;
        font-size: 16px;
        max-width: 650px;
        margin: auto;
        line-height: 1.6;
    }


    /* ========================================================
       TOOL CARDS
       ======================================================== */

    .tool-card {
        background-color: #1a1d26;
        border: 1px solid #2b2f3a;
        border-radius: 12px;
        padding: 13px 15px;
        margin-bottom: 10px;
    }

    .tool-title {
        color: white;
        font-weight: 600;
        font-size: 14px;
    }

    .tool-description {
        color: #8f96a3;
        font-size: 12px;
        margin-top: 3px;
    }


    /* ========================================================
       DOCUMENT
       ======================================================== */

    .document-card {
        background-color: #171a22;
        border: 1px solid #303442;
        border-radius: 12px;
        padding: 14px;
        margin-top: 10px;
    }

    .document-title {
        color: white;
        font-weight: 600;
        font-size: 14px;
    }

    .document-info {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 5px;
    }


    /* ========================================================
       HITL APPROVAL CARD
       ======================================================== */

    .hitl-card {
        background-color: #211f16;
        border: 1px solid #8a742c;
        border-radius: 16px;
        padding: 22px;
        margin: 20px 0;
    }

    .hitl-title {
        color: #f4d35e;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hitl-message {
        color: #e5e7eb;
        font-size: 16px;
        line-height: 1.5;
    }


    /* ========================================================
       CHAT
       ======================================================== */

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 8px;
    }

    div[data-testid="stChatInput"] {
        border-radius: 16px;
    }


    /* ========================================================
       HIDE STREAMLIT BRANDING
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_new_thread_id():

    return str(
        uuid.uuid4()
    )


def get_message_text(
    message,
):

    content = getattr(
        message,
        "content",
        "",
    )


    if isinstance(
        content,
        str,
    ):

        return content


    if isinstance(
        content,
        list,
    ):

        text_parts = []


        for item in content:

            if isinstance(
                item,
                dict,
            ):

                text = item.get(
                    "text"
                )

                if text:

                    text_parts.append(
                        str(text)
                    )


            elif isinstance(
                item,
                str,
            ):

                text_parts.append(
                    item
                )


        return "".join(
            text_parts
        )


    return str(
        content
    )


def get_first_user_message(
    messages,
):

    for message in messages:

        if isinstance(
            message,
            HumanMessage,
        ):

            text = get_message_text(
                message
            ).strip()


            if text:

                return text


    return "New Conversation"


def clean_title(
    text,
    max_length=35,
):

    text = (
        text
        .replace("\n", " ")
        .strip()
    )


    if len(text) <= max_length:

        return text


    return (
        text[:max_length].rstrip()
        + "..."
    )


def display_message(
    message,
):

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    if isinstance(
        message,
        HumanMessage,
    ):

        content = get_message_text(
            message
        )


        if not content.strip():

            return


        with st.chat_message(
            "user"
        ):

            st.markdown(
                content
            )


        return


    # --------------------------------------------------------
    # ASSISTANT
    # --------------------------------------------------------

    if isinstance(
        message,
        AIMessage,
    ):

        content = get_message_text(
            message
        )


        if not content.strip():

            return


        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                content
            )


        return


    # --------------------------------------------------------
    # TOOL MESSAGE
    # --------------------------------------------------------

    if isinstance(
        message,
        ToolMessage,
    ):

        return


def display_conversation(
    messages,
):

    for message in messages:

        display_message(
            message
        )


def save_uploaded_pdf(
    uploaded_file,
):

    upload_directory = "uploads"


    os.makedirs(
        upload_directory,
        exist_ok=True,
    )


    safe_name = os.path.basename(
        uploaded_file.name
    )


    file_path = os.path.join(

        upload_directory,

        safe_name,
    )


    with open(
        file_path,
        "wb",
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    result = ingest_rag_document(
        file_path
    )


    return result


def build_config(
    thread_id,
):

    return {

        "configurable": {

            "thread_id":
            str(thread_id),
        },

        "metadata": {

            "thread_id":
            str(thread_id),
        },

        "run_name":
        "chat_trace",
    }


# ============================================================
# SESSION STATE
# ============================================================

if (
    "current_thread_id"
    not in st.session_state
):

    st.session_state.current_thread_id = (
        make_new_thread_id()
    )


if (
    "uploaded_document"
    not in st.session_state
):

    st.session_state.uploaded_document = None


# ============================================================
# CURRENT THREAD
# ============================================================

current_thread_id = (
    st.session_state.current_thread_id
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-title">'
        '🤖 Agentic AI'
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        '<div class="sidebar-subtitle">'
        'LangGraph + RAG + HITL'
        '</div>',
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋  New Chat",
        use_container_width=True,
    ):

        st.session_state.current_thread_id = (
            make_new_thread_id()
        )

        st.session_state.uploaded_document = (
            None
        )

        st.rerun()


    # --------------------------------------------------------
    # CONVERSATIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'Conversations'
        '</div>',
        unsafe_allow_html=True,
    )


    try:

        threads = get_all_threads()

    except Exception:

        threads = []


    if threads:

        for thread_id in threads:

            try:

                history = get_thread_history(
                    thread_id
                )


                title = clean_title(
                    get_first_user_message(
                        history
                    )
                )


            except Exception:

                title = "Conversation"


            is_current = (
                str(thread_id)
                ==
                str(current_thread_id)
            )


            label = (
                "● "
                if is_current
                else "○ "
            ) + title


            if st.button(
                label,
                key=f"thread_{thread_id}",
                use_container_width=True,
            ):

                st.session_state.current_thread_id = (
                    str(thread_id)
                )

                st.rerun()

    else:

        st.caption(
            "No previous conversations yet."
        )


    # --------------------------------------------------------
    # TOOLS
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'Tools'
        '</div>',
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="tool-card">
            <div class="tool-title">
                🔎 Tavily Search
            </div>
            <div class="tool-description">
                Search the live web
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-title">
                📄 PDF RAG
            </div>
            <div class="tool-description">
                Ask questions about PDFs
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-title">
                🧮 Calculator
            </div>
            <div class="tool-description">
                Perform calculations
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-title">
                📈 Stock
            </div>
            <div class="tool-description">
                Check stock prices
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-title">
                🌤️ Weather
            </div>
            <div class="tool-description">
                Current weather
            </div>
        </div>

        <div class="tool-card">
            <div class="tool-title">
                🛡️ HITL
            </div>
            <div class="tool-description">
                Human approval for stock purchases
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.markdown(
        '<div class="sidebar-section">'
        'Knowledge Base'
        '</div>',
        unsafe_allow_html=True,
    )


    if st.session_state.uploaded_document:

        document = (
            st.session_state.uploaded_document
        )


        st.markdown(
            f"""
            <div class="document-card">

                <div class="document-title">
                    📄 {document["filename"]}
                </div>

                <div class="document-info">
                    {document["pages"]} pages
                    •
                    {document["chunks"]} chunks
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.caption(
            "Attach a PDF using 📎 in the chat box."
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">

        <div class="app-title">
            🤖 Agentic AI Assistant
        </div>

        <div class="app-subtitle">
            OpenRouter • LangGraph • RAG • HITL • Tools
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD CHAT HISTORY
# ============================================================

try:

    saved_messages = get_thread_history(
        current_thread_id
    )

except Exception as error:

    saved_messages = []

    st.error(
        f"Could not load conversation: {error}"
    )


# ============================================================
# DISPLAY WELCOME
# ============================================================

if not saved_messages:

    st.markdown(
        """
        <div class="welcome-container">

            <div class="welcome-icon">
                🤖
            </div>

            <div class="welcome-title">
                How can I help you?
            </div>

            <div class="welcome-description">
                Ask questions, search the web,
                calculate, check stocks and weather,
                upload a PDF, or request a stock purchase.
                Stock purchases require human approval.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


else:

    display_conversation(
        saved_messages
    )


# ============================================================
# CHECK FOR HUMAN APPROVAL
# ============================================================

pending_interrupt = None


try:

    pending_interrupt = get_pending_interrupt(
        current_thread_id
    )

except Exception:

    pending_interrupt = None


# ============================================================
# HUMAN APPROVAL UI
# ============================================================

if pending_interrupt:

    st.markdown(
        """
        <div class="hitl-card">

            <div class="hitl-title">
                ⚠️ Human Approval Required
            </div>

            <div class="hitl-message">
                The AI wants to perform an action
                that requires your approval.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # Extract interrupt information
    # --------------------------------------------------------

    symbol = pending_interrupt.get(
        "symbol",
        "UNKNOWN",
    )


    quantity = pending_interrupt.get(
        "quantity",
        0,
    )


    message = pending_interrupt.get(
        "message",
        f"Approve buying {quantity} shares of {symbol}?",
    )


    # --------------------------------------------------------
    # Display purchase details
    # --------------------------------------------------------

    st.warning(
        f"🛒 **Stock Purchase Request**\n\n"
        f"**Symbol:** `{symbol}`\n\n"
        f"**Quantity:** `{quantity}` shares\n\n"
        f"**Request:** {message}\n\n"
        f"⚠️ This is a **mock purchase** and does "
        f"not connect to a real brokerage."
    )


    # --------------------------------------------------------
    # APPROVE / REJECT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        approve = st.button(
            "✅ Approve Purchase",
            type="primary",
            use_container_width=True,
            key="approve_purchase",
        )


    with col2:

        reject = st.button(
            "❌ Reject Purchase",
            use_container_width=True,
            key="reject_purchase",
        )


    # ========================================================
    # APPROVE
    # ========================================================

    if approve:

        with st.spinner(
            "Processing approved purchase..."
        ):

            try:

                resume_chat(
                    current_thread_id,
                    "yes",
                )


                st.success(
                    "✅ Purchase approved."
                )


                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Could not resume graph: {error}"
                )


    # ========================================================
    # REJECT
    # ========================================================

    if reject:

        with st.spinner(
            "Cancelling purchase..."
        ):

            try:

                resume_chat(
                    current_thread_id,
                    "no",
                )


                st.info(
                    "❌ Purchase rejected."
                )


                st.rerun()


            except Exception as error:

                st.error(
                    f"❌ Could not resume graph: {error}"
                )


# ============================================================
# CHAT INPUT
# ============================================================
#
# Don't show the normal chat input while HITL approval
# is waiting.
#
# This prevents the user from starting another graph
# execution while the current graph is paused.
#
# ============================================================

if not pending_interrupt:

    prompt = st.chat_input(

        "Ask anything, search the web, calculate, check stocks, weather, or ask about your PDF...",

        accept_file=True,

        file_type=["pdf"],

        key="main_chat_input",
    )


    # ========================================================
    # HANDLE USER INPUT
    # ========================================================

    if prompt:

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        user_text = ""

        try:

            user_text = prompt.text

        except AttributeError:

            try:

                user_text = prompt.get(
                    "text",
                    "",
                )

            except Exception:

                user_text = ""


        if user_text is None:

            user_text = ""


        user_text = user_text.strip()


        # ----------------------------------------------------
        # Extract files
        # ----------------------------------------------------

        uploaded_files = []


        try:

            uploaded_files = prompt.files

        except AttributeError:

            try:

                uploaded_files = prompt.get(
                    "files",
                    [],
                )

            except Exception:

                uploaded_files = []


        # ====================================================
        # PROCESS PDF
        # ====================================================

        if uploaded_files:

            uploaded_file = uploaded_files[0]


            try:

                with st.status(
                    "📄 Processing PDF...",
                    expanded=True,
                ) as status:

                    st.write(
                        f"Uploading **{uploaded_file.name}**..."
                    )


                    result = save_uploaded_pdf(
                        uploaded_file
                    )


                    st.session_state.uploaded_document = {

                        "filename":
                        result["filename"],

                        "pages":
                        result["pages"],

                        "chunks":
                        result["chunks"],
                    }


                    status.update(

                        label=
                        "✅ PDF indexed successfully",

                        state="complete",

                        expanded=False,
                    )


            except Exception as error:

                st.error(
                    f"❌ PDF processing failed: {error}"
                )

                st.stop()


        # ====================================================
        # PDF ONLY
        # ====================================================

        if not user_text:

            if uploaded_files:

                st.success(
                    f"📄 **{uploaded_files[0].name}** "
                    "is ready. Ask me anything about it."
                )

            st.rerun()


        # ====================================================
        # DISPLAY USER MESSAGE
        # ====================================================

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_text
            )


            if uploaded_files:

                st.caption(
                    f"📎 {uploaded_files[0].name}"
                )


        # ====================================================
        # RUN LANGGRAPH
        # ====================================================

        config = build_config(
            current_thread_id
        )


        try:

            with st.spinner(
                "🤖 Agent is thinking..."
            ):

                interrupted = False


                # ------------------------------------------------
                # Use updates mode because we need to detect
                # LangGraph's __interrupt__ event.
                # ------------------------------------------------

                for event in chatbot.stream(

                    {

                        "messages": [

                            HumanMessage(
                                content=user_text
                            )
                        ]
                    },

                    config=config,

                    stream_mode="updates",
                ):

                    # --------------------------------------------
                    # Detect HITL interruption
                    # --------------------------------------------

                    if "__interrupt__" in event:

                        interrupted = True

                        break


                # ------------------------------------------------
                # If HITL interrupted the graph:
                #
                # Rerun Streamlit.
                #
                # The checkpoint now contains the pending
                # interrupt, which the UI will display.
                # ------------------------------------------------

                if interrupted:

                    st.rerun()


                # ------------------------------------------------
                # Normal completed response
                # ------------------------------------------------

                st.rerun()


        except Exception as error:

            st.error(
                f"❌ Chat error: {error}"
            )