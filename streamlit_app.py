"""
Streamlit Frontend for Production RAG System
Features: Query, Document Upload, Metrics, System Health
"""

import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page Configuration
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.5rem;
        margin: 0.8rem 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
        line-height: 1.6;
    }
    .source-box strong {
        color: #1f77b4;
        font-weight: 600;
    }
    .source-text {
        color: #34495e;
        font-size: 0.95rem;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #ffffff;
        border-radius: 0.3rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=RAG+System", use_container_width=True)
    st.title("⚙️ Settings")
    
    # Query Settings
    st.subheader("Query Settings")
    top_k = st.slider("Number of Results", 1, 20, 5)
    include_sources = st.checkbox("Include Sources", value=True)
    enable_reranking = st.checkbox("Enable Reranking", value=True)
    
    st.divider()
    
    # Navigation
    st.subheader("📍 Navigation")
    page = st.radio(
        "Select Page",
        ["💬 Chat", "📤 Upload Documents", "📊 Metrics", "🏥 System Health"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Quick Actions
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.conversation_id = None
        st.rerun()
    
    if st.button("🔄 Reset Conversation"):
        st.session_state.conversation_id = None
        st.success("Conversation reset!")

# Helper Functions
def query_api(query: str, top_k: int, include_sources: bool):
    """Send query to API"""
    try:
        payload = {
            "query": query,
            "top_k": top_k,
            "include_sources": include_sources
        }
        
        # Add session_id if exists
        if st.session_state.conversation_id:
            payload["session_id"] = st.session_state.conversation_id
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def upload_document(file, metadata: dict):
    """Upload document to API"""
    try:
        # Reset file pointer to beginning
        file.seek(0)
        
        # Send file only, metadata is optional and handled by API
        files = {"file": (file.name, file, file.type)}
        
        # If metadata is provided, send it as JSON in the request
        # But for now, let's just upload without metadata to test
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload Error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            st.error(f"Details: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None

def get_metrics():
    """Get system metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Metrics Error: {str(e)}")
        return None

def get_health():
    """Get system health"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Health Check Error: {str(e)}")
        return None

# Main Content
st.markdown('<div class="main-header">🤖 Advanced RAG System</div>', unsafe_allow_html=True)

# Page: Chat
if page == "💬 Chat":
    st.subheader("💬 Ask Questions")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(chat["query"])
        
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            
            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{chat.get('confidence_score', 0):.2%}")
            with col2:
                st.metric("Query Type", chat.get('query_type', 'N/A'))
            with col3:
                st.metric("Processing Time", f"{chat.get('processing_time_ms', 0):.0f}ms")
            
            # Show sources
            if chat.get('citations'):
                with st.expander(f"📚 View {len(chat['citations'])} Sources"):
                    for idx, citation in enumerate(chat['citations'], 1):
                        text_snippet = citation.get('text_snippet', citation.get('text', 'N/A'))
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {idx}:</strong> {citation.get('source', 'Unknown')}<br>
                            <strong>Relevance:</strong> {citation.get('relevance_score', 0):.2%}<br>
                            <div class="source-text">{text_snippet[:400]}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Query input
    query = st.chat_input("Ask a question...")
    
    if query:
        # Add user message
        with st.chat_message("user"):
            st.write(query)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = query_api(query, top_k, include_sources)
                
                if result:
                    st.write(result.get('answer', 'No answer received'))
                    
                    # Update session ID
                    if result.get('session_id'):
                        st.session_state.conversation_id = result['session_id']
                    
                    # Show metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", f"{result.get('confidence_score', 0):.2%}")
                    with col2:
                        st.metric("Query Type", result.get('query_type', 'N/A'))
                    with col3:
                        st.metric("Processing Time", f"{result.get('processing_time_ms', 0):.0f}ms")
                    
                    # Show sources
                    if result.get('citations'):
                        with st.expander(f"📚 View {len(result['citations'])} Sources"):
                            for idx, citation in enumerate(result['citations'], 1):
                                text_snippet = citation.get('text_snippet', citation.get('text', 'N/A'))
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {idx}:</strong> {citation.get('source', 'Unknown')}<br>
                                    <strong>Relevance:</strong> {citation.get('relevance_score', 0):.2%}<br>
                                    <div class="source-text">{text_snippet[:400]}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": result.get('answer'),
                        "confidence_score": result.get('confidence_score'),
                        "query_type": result.get('query_type'),
                        "processing_time_ms": result.get('processing_time_ms'),
                        "citations": result.get('citations', [])
                    })
                    
                    # Feedback buttons
                    st.divider()
                    col1, col2, col3 = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍 Helpful", key=f"helpful_{len(st.session_state.chat_history)}"):
                            try:
                                feedback_response = requests.post(
                                    f"{API_BASE_URL}/feedback",
                                    json={
                                        "query": query,
                                        "answer": result.get('answer'),
                                        "rating": 5,
                                        "helpful": True,
                                        "comment": "Marked as helpful"
                                    }
                                )
                                if feedback_response.status_code == 200:
                                    st.success("Thanks for your feedback! 👍")
                            except:
                                st.info("Feedback recorded locally")
                    
                    with col2:
                        if st.button("👎 Not Helpful", key=f"not_helpful_{len(st.session_state.chat_history)}"):
                            try:
                                feedback_response = requests.post(
                                    f"{API_BASE_URL}/feedback",
                                    json={
                                        "query": query,
                                        "answer": result.get('answer'),
                                        "rating": 1,
                                        "helpful": False,
                                        "comment": "Marked as not helpful"
                                    }
                                )
                                if feedback_response.status_code == 200:
                                    st.warning("Thanks for your feedback. We'll improve! 👎")
                            except:
                                st.info("Feedback recorded locally")

# Page: Upload Documents
elif page == "📤 Upload Documents":
    st.subheader("📤 Upload Documents")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx', 'md'],
            help="Upload PDF, TXT, DOCX, or MD files"
        )
    
    with col2:
        st.info("📝 Supported formats:\n- PDF\n- TXT\n- DOCX\n- Markdown")
    
    # Metadata
    st.subheader("📋 Document Metadata (Optional)")
    col1, col2 = st.columns(2)
    
    with col1:
        doc_title = st.text_input("Title")
        doc_category = st.selectbox("Category", ["Technical", "Business", "General", "Other"])
    
    with col2:
        doc_author = st.text_input("Author")
        doc_tags = st.text_input("Tags (comma-separated)")
    
    if st.button("📤 Upload Document", type="primary", disabled=not uploaded_file):
        if uploaded_file:
            metadata = {
                "title": doc_title or uploaded_file.name,
                "category": doc_category,
                "author": doc_author,
                "tags": [tag.strip() for tag in doc_tags.split(",")] if doc_tags else [],
                "upload_date": datetime.now().isoformat()
            }
            
            with st.spinner("Uploading and processing..."):
                result = upload_document(uploaded_file, metadata)
                
                if result:
                    st.success(f"✅ Document uploaded successfully!")
                    st.json(result)

# Page: Metrics
elif page == "📊 Metrics":
    st.subheader("📊 System Metrics")
    
    metrics = get_metrics()
    
    if metrics:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", metrics.get('total_queries', 0))
        with col2:
            st.metric("Avg Response Time", f"{metrics.get('average_response_time_ms', 0):.0f}ms")
        with col3:
            st.metric("Cache Hit Rate", f"{metrics.get('cache_hit_rate', 0):.1%}")
        with col4:
            st.metric("Error Rate", f"{metrics.get('error_rate', 0):.1%}")
        
        st.divider()
        
        # Additional Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Performance")
            st.metric("Uptime", f"{metrics.get('uptime_seconds', 0) / 3600:.1f} hours")
            st.metric("Total Errors", metrics.get('total_errors', 0))
        
        with col2:
            st.subheader("💾 Cache Statistics")
            st.metric("Cache Hits", metrics.get('cache_hits', 0))
            st.metric("Cache Misses", metrics.get('cache_misses', 0))
        
        # Query Types Distribution
        if metrics.get('query_types'):
            st.subheader("📈 Query Types Distribution")
            df = pd.DataFrame(
                list(metrics['query_types'].items()),
                columns=['Query Type', 'Count']
            )
            st.bar_chart(df.set_index('Query Type'))

# Page: System Health
elif page == "🏥 System Health":
    st.subheader("🏥 System Health")
    
    health = get_health()
    
    if health:
        # Overall Status
        status = health.get('status', 'unknown')
        if status == 'healthy':
            st.success("✅ System is healthy")
        else:
            st.error("❌ System has issues")
        
        st.divider()
        
        # Component Status
        col1, col2, col3 = st.columns(3)
        
        components = health.get('components', {})
        
        with col1:
            st.subheader("🗄️ Vector Store")
            vector_status = components.get('vector_store', 'unknown')
            if vector_status == 'healthy':
                st.success("✅ Healthy")
            else:
                st.error("❌ Unhealthy")
        
        with col2:
            st.subheader("💾 Cache")
            cache_status = components.get('cache', 'unknown')
            if cache_status == 'healthy':
                st.success("✅ Healthy")
            else:
                st.error("❌ Unhealthy")
        
        with col3:
            st.subheader("🤖 LLM")
            llm_status = components.get('llm', 'unknown')
            if llm_status == 'healthy':
                st.success("✅ Healthy")
            else:
                st.error("❌ Unhealthy")
        
        st.divider()
        
        # System Info
        st.subheader("ℹ️ System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Version:** {health.get('version', 'N/A')}")
            st.write(f"**Environment:** {health.get('environment', 'N/A')}")
        
        with col2:
            st.write(f"**Timestamp:** {health.get('timestamp', 'N/A')}")
            st.write(f"**Uptime:** {health.get('uptime_seconds', 0) / 3600:.1f} hours")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Advanced RAG System | Built with FastAPI + LangGraph + Qdrant</p>
    <p>Made with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Made with Bob
