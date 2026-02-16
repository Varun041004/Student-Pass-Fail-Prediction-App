import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🎓 Student ML System",
    layout="wide",
    page_icon="📊"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.block-container {
    padding-top: 2rem;
}
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
div.stButton > button:hover {
    background-color: #45a049;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
def login():
    st.markdown("## 🔐 Login to Student ML System")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "1234":
                st.session_state.logged_in = True
                st.success("Login successful 🎉")
                st.rerun()
            else:
                st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD DATA + MODEL ----------------
df = pd.read_excel("student_data.xlsx")
df["Result"] = df["Result"].map({"Pass": 1, "Fail": 0})

X = df.drop("Result", axis=1)
y = df["Result"]

model = joblib.load("model.pkl")

pred = model.predict(X)
accuracy = accuracy_score(y, pred)

# ---------------- SIDEBAR ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=100)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("", ["🏠 Prediction", "📊 Dashboard", "🤖 About Model"])

st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- PREDICTION PAGE ----------------
if page == "🏠 Prediction":
    st.title("🎓 Student Performance Prediction")

    st.markdown("### Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider("📚 Study Hours", 0, 10, 0)
        attendance = st.slider("📝 Attendance (%)", 0, 100, 0)
        assignments = st.slider("📂 Assignments Completed", 0, 10, 0)

    with col2:
        internal_marks = st.slider("📊 Internal Marks", 0, 100, 0)
        sleep_hours = st.slider("😴 Sleep Hours", 0, 9, 0)


    st.markdown("---")

    if st.button("🔍 Predict Result"):
        new_student = pd.DataFrame(
            [[study_hours, attendance, assignments, internal_marks, sleep_hours]],
            columns=X.columns
        )

        result = model.predict(new_student)

        if result[0] == 1:
            st.success("✅ Student is likely to PASS")
        else:
            st.error("❌ Student is likely to FAIL")

# ---------------- DASHBOARD PAGE ----------------
elif page == "📊 Dashboard":
    st.title("📊 Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", len(df))
    col2.metric("Pass %", f"{(df['Result'].mean()*100):.1f}%")
    col3.metric("Model Accuracy", f"{accuracy*100:.2f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pass vs Fail Distribution")
        fig1, ax1 = plt.subplots()
        sns.countplot(x="Result", data=df, ax=ax1)
        ax1.set_xticklabels(["Fail", "Pass"])
        st.pyplot(fig1)

    with col2:
        st.subheader("Average Study Hours by Result")
        fig2, ax2 = plt.subplots()
        df.groupby("Result")["StudyHours"].mean().plot(kind="bar", ax=ax2)
        ax2.set_xticklabels(["Fail", "Pass"], rotation=0)
        st.pyplot(fig2)

    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# ---------------- ABOUT MODEL PAGE ----------------
elif page == "🤖 About Model":
    st.title("🤖 About Machine Learning Model")

    st.info("""
    **Model Used:** Decision Tree Classifier  
    **Problem Type:** Classification  
    **Goal:** Predict student performance (Pass/Fail)
    """)

    st.subheader("📈 Model Performance")
    st.success(f"Accuracy: {accuracy*100:.2f}%")

    st.subheader("📂 Dataset Details")
    st.write(f"Number of records: {len(df)}")
    st.write("Features Used:")
    st.write(list(X.columns))

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>💡 For such kinda webapps - Contact: <b>my_mail</b></center>",
    unsafe_allow_html=True
)
