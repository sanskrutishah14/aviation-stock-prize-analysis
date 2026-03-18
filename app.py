import streamlit as st
import base64

st.set_page_config(
    page_title="Airline Forecast Dashboard",
    page_icon="✈️",
    layout="wide"
)
#styles
st.markdown("""
    <style>
        .stApp {
            /* Sky-blue background */
            background: linear-gradient(180deg, #e9f4ff 0%, #d4ecff 60%, #c8e6ff 100%);
            color: #212529;
            font-family: 'Poppins', sans-serif;
        }
        h1, h2, h3, h4 {
            color: #0d1b2a !important;
            font-weight: 600;
            text-align: center;
        }
        .airline-card {
            background: rgba(255, 255, 255, 0.97);
            border-radius: 18px;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
            margin-bottom: 14px; /* snug so the button hugs the card */
        }
        .airline-card:hover {
            transform: translateY(-4px);
            box-shadow: 0px 10px 22px rgba(0,0,0,0.15);
        }
        .airline-logo {
            width: 150px;
            height: 150px;
            object-fit: contain;
            margin-bottom: 12px;
            border-radius: 14px;
            background-color: white;
            padding: 10px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.08);
        }

        /* Center ALL buttons in any Streamlit column */
        [data-testid="column"] div[data-testid="stButton"] > button {
            display: block;
            margin: 10px auto 28px auto;   /* centers horizontally */
            text-align: center;
        }

        /* Keep your button visuals */
        div[data-testid="stButton"] > button {
            background: linear-gradient(145deg, #212529, #343a40);
            color: #f8f9fa;
            border-radius: 14px;
            font-size: 17px;
            font-weight: 600;
            padding: 0.8em 2em;
            transition: all 0.2s ease-in-out;
            box-shadow: 0px 6px 14px rgba(0,0,0,0.22);
            border: none;
        }
        div[data-testid="stButton"] > button:hover {
            background: linear-gradient(145deg, #000000, #343a40);
            color: #00b4d8;
            transform: translateY(-2px);
            box-shadow: 0px 10px 22px rgba(0,0,0,0.28);
        }

        /* Bottom action row: ensure buttons are centered INSIDE each half */
        .bottom-row [data-testid="column"] {
            display: flex; 
            flex-direction: column; 
            align-items: center;      /* centers children in each half */
        }
        .bottom-row div[data-testid="stButton"] > button {
            width: 260px;             /* consistent width for optical centering */
            margin: 6px auto 0 auto;  /* perfect center */
        }
    </style>
""", unsafe_allow_html=True)


#front page
st.title("✈️ Airline Forecasting and Sensitivity Dashboard")
st.markdown("### Choose an airline to explore predictions and insights.")
st.markdown("<hr>", unsafe_allow_html=True)

#airline data
airlines = {
    "American Airlines": ("assets/american-airlines-logo.png.png", "american"),
    "Delta Airlines": ("assets/Delta-Air-Lines-Logo.png", "delta"),
    "United Airlines": ("assets/United_Airlines_Logo.svg.png", "united"),
    "Southwest Airlines": ("assets/Southwest_Airlines_logo_2014.svg.png", "southwest")
}

#grid
cols = st.columns(4)

for i, (name, (img_path, page)) in enumerate(airlines.items()):
    with cols[i]:
        try:
            with open(img_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode()
            st.markdown(f"""
            <div class="airline-card">
                <img src="data:image/png;base64,{encoded}" class="airline-logo">
                <h4>{name}</h4>
            </div>
            """, unsafe_allow_html=True)
        except Exception:
            st.error(f"⚠️ Could not load {img_path}")

        #select airline button
        if st.button(f"Select {name}", key=name):
            st.session_state["selected_airline"] = name
            st.success(f"{name} selected!")

#next step button
st.markdown("<hr>", unsafe_allow_html=True)

if "selected_airline" in st.session_state:
    st.subheader(f"Selected Airline: {st.session_state['selected_airline']}")
    # Wrap the two columns so CSS can center buttons inside each half
    st.markdown('<div class="bottom-row">', unsafe_allow_html=True)
    # Use a 5-column layout so the two action buttons are equidistant from the page center
    spacer_left, col_left, spacer_mid, col_right, spacer_right = st.columns([1, 2, 1, 2, 1])
    with col_left:
        if st.button("Use one model"):
            st.switch_page("pages/one_model.py")
    with col_right:
        if st.button("Compare models"):
            st.switch_page("pages/compare_models.py")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Select an airline first to continue.")
