import os
import streamlit as st
from typing import List
import pandas as pd
import plotly.express as px
import sys
from transformers import pipeline

# Initial prints for debugging versions
print(f"Python version: {sys.version}")

# Load the Hugging Face API token from the environment variable
api_token = os.getenv("huggingface_api_token")

# Check if the API token is available
if api_token is None:
    st.error("Hugging Face API token not found. Please set the huggingface_api_token environment variable.")
else:
    try:
        # Load the model from Hugging Face using the API token
        model = pipeline("text-generation", model="najii/llama-guard", use_auth_token=api_token)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Define your support functions
def create_pagination(df: pd.DataFrame, items_per_page: int, label: str):
    start_idx = st.session_state.page_number * items_per_page
    end_idx = start_idx + items_per_page
    return df.iloc[start_idx:end_idx]

def generate_grid_gallery(df_subset: pd.DataFrame, n_cols=5, extra_key=""):
    for index, row in df_subset.iterrows():
        st.write(row)

def create_bottom_navigation(label: str):
    if st.button("Previous Page", key=f"prev_{label}"):
        st.session_state.page_number = max(0, st.session_state.page_number - 1)
    if st.button("Next Page", key=f"next_{label}"):
        st.session_state.page_number += 1

def plot_publication_counts(df: pd.DataFrame, cumulative: bool):
    if cumulative:
        df = df.cumsum()
    fig = px.line(df, x=df.index, y="count", title="Publication Counts")
    return fig

# Initialize session state for pagination
if "page_number" not in st.session_state:
    st.session_state.page_number = 0

# Load or define your DataFrame here
# Example: papers_df = pd.read_csv("path_to_your_papers_data.csv")

# Create tabs for navigating different sections
content_tabs = st.tabs(["Grid View", "Publication Count", "Focus on Paper", "Chat with GPT", "Repositories", "Weekly Review"])

with content_tabs[0]:
    # Page for grid view
    papers_df_subset = create_pagination(papers_df, items_per_page=25, label="grid")
    generate_grid_gallery(papers_df_subset)
    create_bottom_navigation(label="grid")

with content_tabs[1]:
    # Page for publication count
    total_papers = len(papers_df)
    st.markdown(f"### 📈 Total publication counts: {total_papers}")
    plot_type = st.radio(
        label="Plot type",
        options=["Daily", "Cumulative"],
        index=1,
        label_visibility="collapsed",
        horizontal=True,
    )
    cumulative = plot_type == "Cumulative"
    ts_plot = plot_publication_counts(papers_df, cumulative=cumulative)
    st.plotly_chart(ts_plot, use_container_width=True)

    # Assuming the existence of a `plot_cluster_map` function
    st.markdown(f"### Topic Model Map")
    cluster_map = plot_cluster_map(papers_df)
    st.plotly_chart(cluster_map, use_container_width=True)

with content_tabs[2]:
    # Focus on a paper
    arxiv_code = st.text_input("Arxiv Code", "")
    st.session_state.arxiv_code = arxiv_code
    if len(arxiv_code) > 0:
        if arxiv_code in full_papers_df.index:
            paper = full_papers_df.loc[arxiv_code].to_dict()
            generate_grid_gallery(pd.DataFrame([paper]))
        else:
            st.error("Paper not found.")

with content_tabs[3]:
    # Chat with GPT
    st.markdown("##### 🤖 Chat with the GPT Maestro.")
    user_question = st.text_area("Ask any question about LLMs or the ArXiv papers.", value="")
    chat_btn_disabled = len(user_question) == 0
    chat_cols = st.columns((1, 2, 1))
    chat_btn = chat_cols[0].button("Send", disabled=chat_btn_disabled)
    response_length = "short answer"
    if chat_btn:
        if user_question != "":
            with st.spinner("Consulting the GPT Maestro, this might take a minute..."):
                response, referenced_codes, relevant_codes = query_llmpedia_new(user_question, response_length)
                st.divider()
                st.markdown(response)
                if referenced_codes:
                    st.divider()
                    st.markdown("<h4>Referenced Papers:</h4>", unsafe_allow_html=True)
                    reference_df = st.session_state["papers"].loc[referenced_codes]
                    generate_grid_gallery(reference_df, n_cols=5, extra_key="_chat")
                if relevant_codes:
                    st.divider()
                    st.markdown("<h4>Other Relevant Papers:</h4>", unsafe_allow_html=True)
                    relevant_df = st.session_state["papers"].loc[relevant_codes]
                    generate_grid_gallery(relevant_df, n_cols=5, extra_key="_chat")

with content_tabs[4]:
    # Repositories
    repos_df = pd.DataFrame({
        "topic": ["nlp", "cv"],
        "domain": ["ai", "ml"],
        "title": ["repo 1", "repo 2"],
        "published": [2022, 2023],
        "repo_url": ["http://example.com/1", "http://example.com/2"],
        "repo_title": ["Repo Title 1", "Repo Title 2"],
        "repo_description": ["Desc 1", "Desc 2"],
    })
    repos_search_cols = st.columns((1, 1, 1))
    topic_filter = repos_search_cols[0].multiselect("Filter by Topic", options=repos_df["topic"].unique().tolist(), default=[])
    domain_filter = repos_search_cols[1].multiselect("Filter by Domain", options=repos_df["domain"].value_counts().index.tolist(), default=[])
    search_term = repos_search_cols[2].text_input("Search by Title", value="")

    def filter_repos(df: pd.DataFrame, search_term: str, topic_filter: List[str], domain_filter: List[str]):
        if search_term:
            df = df[df["title"].str.contains(search_term, case=False)]
        if topic_filter:
            df = df[df["topic"].isin(topic_filter)]
        if domain_filter:
            df = df[df["domain"].isin(domain_filter)]
        return df

    filtered_repos = filter_repos(repos_df, search_term, topic_filter, domain_filter)
    repo_count = len(filtered_repos)
    repos_title = f"### 📦 Total resources found: {repo_count}"
    st.markdown(repos_title)
    st.data_editor(
        filtered_repos.drop(columns=["published"]).sort_index(ascending=False),
        column_config={
            "topic": st.column_config.ListColumn("Topic", width="medium"),
            "domain": st.column_config.ListColumn("Domain", width="medium"),
            "repo_url": st.column_config.LinkColumn("Repository URL"),
            "repo_title": st.column_config.TextColumn("Repository Title", width="medium"),
            "repo_description": st.column_config.TextColumn("Repository Description"),
        },
        disabled=True,
    )

    plot_by = st.selectbox("Plot total resources by", options=["Topic", "Domain", "Published"], index=0)

    if filtered_repos.shape[0] > 0:
        plot_repos = plot_repos_by_feature(filtered_repos, plot_by)
        st.plotly_chart(plot_repos, use_container_width=True)

with content_tabs[5]:
    # Weekly Review
    weekly_plot_container = st.empty()
    report_top_cols = st.columns((5, 2))
    with report_top_cols[0]:
        week_reported = st.date_input("Select a week", max_value=get_max_report_date(), value=get_max_report_date())
    with report_top_cols[1]:
        st.markdown("🗓️")
        initialize_report = st.button("🔄 Refresh Weekly Review")
        if initialize_report:
            with st.spinner(f"Generating weekly review for {week_reported.strftime('%y-%m-%d')}..."):
                weekly_summary_content, highlight_content, repo_content = initialize_weekly_summary(date_report=week_reported)
                st.session_state.weekly_summary_content = weekly_summary_content
                st.session_state.highlight_content = highlight_content
                st.session_state.repo_content = repo_content

    st.divider()

    # Weekly summary content
    st.markdown("##### 🗓️ Weekly Summary")
    if "weekly_summary_content" in st.session_state:
        st.markdown(st.session_state.weekly_summary_content)

    # Highlight content
    st.markdown("##### 🥇 Highlight of the Week")
    if "highlight_content" in st.session_state:
        st.markdown(st.session_state.highlight_content)

    # Repository content
    st.markdown("##### 📦 Related Repositories")
    if "repo_content" in st.session_state:
        st.markdown(st.session_state.repo_content)

# Error handling example
try:
    main()  # Ensure `main()` or a relevant function is defined
except Exception as e:
    st.error(f"Error: {str(e)}")
