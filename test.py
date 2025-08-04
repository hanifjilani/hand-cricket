import streamlit as st

st.markdown("""
    <style>
    [data-testid="stColumn"] {
        background-color: #ffffffcc; /* semi-transparent white */
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        margin-top: 10px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    [data-testid="stColumn"]:hover {
        transform: scale(1.02); /* subtle hover effect */
    }
    
    </style>
    """, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3,border=True)

with col1:
    st.write("This is column 1")
with col2:
    st.write("This is column 2 with a blue background")
with col3:
    st.write("This is column 3")