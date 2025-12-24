def run_rsi_divergences_app():
    st.title("📈 RSI Divergences")

    # Add the requested link and description above the HTML import
    st.markdown("""
        **Note:** You can view divergences of the S&P 100, NQ 100 and Mr Darcy's shitco list here: 
        https://mr-darcys-rsi-divergence.streamlit.app/
    """)
    st.markdown("---") # Visual separator

    folder_id = st.secrets.get("GDRIVE_FOLDER_ID")
    api_key = st.secrets.get("GDRIVE_API_KEY")
    if folder_id and api_key:
        try:
            query = f"'{folder_id}'+in+parents+and+name+contains+'.html'+and+trashed=false"
            url = f"https://www.googleapis.com/drive/v3/files?q={query}&orderBy=name+desc&pageSize=1&key={api_key}"
            response_obj = requests.get(url, timeout=10)
            if response_obj.status_code != 200:
                st.error(f"Google Drive API Error ({response_obj.status_code}): {response_obj.text}")
                return
            response = response_obj.json()
            files = response.get('files', [])
            if not files:
                st.warning("No HTML files found in the specified Google Drive folder.")
                return
            latest_id = files[0]['id']
            latest_name = files[0]['name']
            content_url = f"https://www.googleapis.com/drive/v3/files/{latest_id}?alt=media&key={api_key}"
            content_response = requests.get(content_url, timeout=10)
            if content_response.status_code != 200:
                st.error(f"Failed to download file content.")
                return
            content_response.encoding = 'utf-8'
            html_content = content_response.text
            st.info(f"Loaded: {latest_name}")
            components.html(html_content, height=1200, scrolling=True)
            return
        except Exception as e:
            st.error(f"Cloud load execution failed: {e}.")
    else:
        st.error("Google Drive Secrets (GDRIVE_FOLDER_ID / GDRIVE_API_KEY) are missing.")
