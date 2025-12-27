import streamlit as st
import pandas as pd
import random
import time

# -----------------------------------------------------------------------------
# CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mr. Darcy's New Manor",
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to give it a Regency flair
st.markdown("""
<style>
    .main {
        background-color: #fcfbf4;
        color: #2c3e50;
    }
    h1, h2, h3 {
        font-family: 'Georgia', serif;
        color: #1a237e;
    }
    .stButton>button {
        background-color: #1a237e;
        color: white;
        font-family: 'Georgia', serif;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #283593;
        border-color: #283593;
    }
    .stProgress .st-bo {
        background-color: #1a237e;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Georgia', serif;
        color: #b71c1c;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# STATE MANAGEMENT
# -----------------------------------------------------------------------------
if 'budget' not in st.session_state:
    st.session_state.budget = 50000  # Pounds Sterling
if 'reputation' not in st.session_state:
    st.session_state.reputation = 50
if 'renovations' not in st.session_state:
    st.session_state.renovations = []
if 'guests' not in st.session_state:
    st.session_state.guests = [
        {"Name": "Elizabeth Bennet", "Status": "Invited", "Relation": "Complicated"},
        {"Name": "Charles Bingley", "Status": "Invited", "Relation": "Best Friend"},
        {"Name": "Jane Bennet", "Status": "Invited", "Relation": "Friend"},
    ]
if 'letters' not in st.session_state:
    st.session_state.letters = []

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def format_currency(amount):
    return f"£{amount:,}"

def add_renovation(name, cost, rep_gain):
    if st.session_state.budget >= cost:
        st.session_state.budget -= cost
        st.session_state.reputation = min(100, st.session_state.reputation + rep_gain)
        st.session_state.renovations.append(name)
        return True, f"Successfully renovated the {name}!"
    else:
        return False, "Insufficient funds, sir. We must economize."

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Pemberley_Chatsworth_House.jpg/640px-Pemberley_Chatsworth_House.jpg", caption="The New Estate")
st.sidebar.title("The Steward's Ledger")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Estate Overview", "Renovations", "Guest List", "Correspondence"])

st.sidebar.markdown("---")
st.sidebar.header("Current Standing")
st.sidebar.metric("Estate Funds", format_currency(st.session_state.budget))
st.sidebar.metric("Social Reputation", f"{st.session_state.reputation}/100")

if st.session_state.reputation < 30:
    st.sidebar.warning("Warning: Lady Catherine is displeased.")
elif st.session_state.reputation > 80:
    st.sidebar.success("The ton is abuzz with your success!")

# -----------------------------------------------------------------------------
# PAGE: ESTATE OVERVIEW
# -----------------------------------------------------------------------------
if menu == "Estate Overview":
    st.title("🏰 Welcome to the New Manor")
    st.markdown("*A fine morning to you, Mr. Darcy. The estate requires your immediate attention.*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Estate Status")
        st.write("The grounds are extensive, though the East Wing is drafty and the gardens are currently overgrown. We must prepare for the upcoming ball.")
        
        if not st.session_state.renovations:
            st.info("No renovations have been completed yet.")
        else:
            st.success("Completed Renovations:")
            for item in st.session_state.renovations:
                st.write(f"- ✅ {item}")

    with col2:
        st.subheader("Lady Catherine's Disapproval Meter")
        # Logic: Higher reputation actually lowers her disapproval usually, 
        # but let's say she hates change, so more renovations = more disapproval.
        disapproval = min(100, len(st.session_state.renovations) * 20)
        st.progress(disapproval / 100)
        if disapproval > 80:
            st.error("She has written three angry letters this morning.")
        elif disapproval > 40:
            st.warning("She is muttering about 'shades of Pemberley'.")
        else:
            st.info("She is currently ignoring us. A blessing.")

# -----------------------------------------------------------------------------
# PAGE: RENOVATIONS
# -----------------------------------------------------------------------------
elif menu == "Renovations":
    st.title("🛠️ Estate Renovations")
    st.write("Where shall we allocate funds to impress Miss Elizabeth?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://images.unsplash.com/photo-1558591710-4b4a1ae0f04d?auto=format&fit=crop&q=80&w=300&h=200", caption="The Gardens")
        st.subheader("Landscape Gardens")
        st.write("Cost: £5,000")
        st.write("Reputation: +15")
        if "Landscape Gardens" in st.session_state.renovations:
            st.button("Completed", disabled=True, key="btn_gardens")
        else:
            if st.button("Renovate Gardens"):
                success, msg = add_renovation("Landscape Gardens", 5000, 15)
                if success: st.success(msg)
                else: st.error(msg)

    with col2:
        st.image("https://images.unsplash.com/photo-1519167758481-83f55a94950d?auto=format&fit=crop&q=80&w=300&h=200", caption="The Ballroom")
        st.subheader("Gilded Ballroom")
        st.write("Cost: £15,000")
        st.write("Reputation: +30")
        if "Gilded Ballroom" in st.session_state.renovations:
            st.button("Completed", disabled=True, key="btn_ballroom")
        else:
            if st.button("Renovate Ballroom"):
                success, msg = add_renovation("Gilded Ballroom", 15000, 30)
                if success: st.success(msg)
                else: st.error(msg)

    with col3:
        st.image("https://images.unsplash.com/photo-1541963463532-d68292c34b19?auto=format&fit=crop&q=80&w=300&h=200", caption="The Library")
        st.subheader("Expand Library")
        st.write("Cost: £8,000")
        st.write("Reputation: +10")
        if "Expand Library" in st.session_state.renovations:
            st.button("Completed", disabled=True, key="btn_library")
        else:
            if st.button("Renovate Library"):
                success, msg = add_renovation("Expand Library", 8000, 10)
                if success: st.success(msg)
                else: st.error(msg)

# -----------------------------------------------------------------------------
# PAGE: GUEST LIST
# -----------------------------------------------------------------------------
elif menu == "Guest List":
    st.title("📜 The Guest List")
    st.write("Manage invitations for the upcoming ball.")

    # Input to add new guest
    with st.form("add_guest"):
        c1, c2, c3 = st.columns(3)
        new_name = c1.text_input("Guest Name")
        new_relation = c2.text_input("Relation")
        submit = st.form_submit_button("Send Invitation")
        
        if submit and new_name:
            if "Wickham" in new_name:
                st.error("Absolutely not. That man is forbidden from the grounds.")
            else:
                st.session_state.guests.append({"Name": new_name, "Status": "Invited", "Relation": new_relation})
                st.success(f"Invitation sent to {new_name}.")

    # Display Guest Table
    df = pd.DataFrame(st.session_state.guests)
    
    # Allow removing guests
    st.subheader("Current Guest List")
    
    # We use data_editor to make it interactive if using newer Streamlit, 
    # but for compatibility, we'll just show the table and a remove selector.
    st.table(df)

    guest_to_remove = st.selectbox("Rescind Invitation from:", ["Select..."] + [g['Name'] for g in st.session_state.guests])
    if guest_to_remove != "Select...":
        if st.button("Remove Guest"):
            st.session_state.guests = [g for g in st.session_state.guests if g['Name'] != guest_to_remove]
            st.experimental_rerun()

# -----------------------------------------------------------------------------
# PAGE: CORRESPONDENCE
# -----------------------------------------------------------------------------
elif menu == "Correspondence":
    st.title("🖋️ Correspondence")
    st.write("Draft a letter. Choose your words carefully.")

    recipient = st.selectbox("To:", ["Elizabeth Bennet", "Georgiana Darcy", "Lady Catherine de Bourgh"])
    tone = st.select_slider("Tone of Letter", options=["Cold", "Formal", "Civil", "Warm", "Ardent"])
    
    message = st.text_area("Body of the letter", height=150, placeholder="My dear...")

    if st.button("Send Letter"):
        if not message:
            st.warning("You cannot send an empty letter, sir.")
        else:
            with st.spinner("The footman is delivering your post..."):
                time.sleep(1.5)
            
            # Simple "AI" response logic
            st.session_state.letters.append(f"To {recipient} ({tone}): {message[:30]}...")
            
            st.markdown("### Reply Received:")
            if recipient == "Elizabeth Bennet":
                if tone == "Ardent":
                    st.success("*She is surprised by your candor, but smiles.*")
                elif tone == "Cold":
                    st.error("*The letter was returned unopened.*")
                else:
                    st.info("*She sends her regards to Georgiana.*")
            elif recipient == "Lady Catherine de Bourgh":
                st.warning("*She demands to know why you are not at Rosings.*")
            elif recipient == "Georgiana Darcy":
                st.success("*She is delighted to hear from her brother.*")

    if st.session_state.letters:
        st.markdown("---")
        st.subheader("Sent Letters Archive")
        for l in st.session_state.letters:
            st.text(l)
