import os
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import streamlit as st

# ------------------------------------------------------------------------------
# 1. Configuration & Initialization
# ------------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY is missing! Add it to your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

st.set_page_config(
    page_title="StudyBuddy AI with Memory",
    page_icon="🧠",
    layout="wide"
)


# Pydantic Schema for Quiz Generator
class Question(BaseModel):
    id: int
    question: str = Field(description="The question prompt")
    options: List[str] = Field(description="List of 4 multiple-choice options")
    correct_option_index: int = Field(description="Zero-based index (0-3) of the correct answer")
    explanation: str = Field(description="Brief explanation of why the correct answer is right")


class Quiz(BaseModel):
    title: str
    questions: List[Question]


# Session State Initialization (Memory Stores)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_memory" not in st.session_state:
    st.session_state.user_memory = {
        "name": "Student",
        "skill_level": "Beginner",
        "notes": "Prefers practical examples over heavy theory."
    }
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# ------------------------------------------------------------------------------
# 2. Sidebar - Long-Term User Memory & Settings
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 User Memory")

    # Editable memory fields
    st.session_state.user_memory["name"] = st.text_input(
        "Name", value=st.session_state.user_memory["name"]
    )

    subject = st.selectbox(
        "Subject Focus",
        ["Computer Science", "Mathematics", "Physics", "General Knowledge"]
    )

    st.session_state.user_memory["skill_level"] = st.select_slider(
        "Skill Level",
        options=["Beginner", "Intermediate", "Advanced"]
    )

    teaching_style = st.select_slider(
        "Teaching Style",
        options=["Simple & Direct", "Socratic (Guide with Questions)", "In-depth Academic"]
    )

    st.session_state.user_memory["notes"] = st.text_area(
        "Learner Notes / Key Context",
        value=st.session_state.user_memory["notes"],
        help="Extra context the AI tutor should remember about you."
    )

    st.markdown("---")
    if st.button("🗑️ Reset Chat History"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------------------
# 3. Main Navigation
# ------------------------------------------------------------------------------
tab_chat, tab_quiz = st.tabs(["💬 Chat Tutor", "📝 Quiz Generator"])


# Dynamic System Instruction incorporating User Memory
def build_system_instruction():
    mem = st.session_state.user_memory
    return f"""
You are StudyBuddy, an empathetic and highly competent AI tutor.

User Context & Memory:
- Name: {mem['name']}
- Subject Focus: {subject}
- Experience Level: {mem['skill_level']}
- Teaching Style Preferred: {teaching_style}
- Specific Student Context/Notes: {mem['notes']}

Guidelines:
- Address the student naturally and adapt all explanations to their stated experience level ({mem['skill_level']}).
- Keep track of concepts discussed previously in the conversation.
- Use clean formatting with bolding, bullet points, and code blocks.
"""


# ------------------------------------------------------------------------------
# TAB 1: Chat Tutor (Short-Term & Long-Term Memory)
# ------------------------------------------------------------------------------
with tab_chat:
    st.title("📚 StudyBuddy Chat")
    st.caption(
        f"Personalized for **{st.session_state.user_memory['name']}** ({st.session_state.user_memory['skill_level']})")

    # Display full chat history from session memory
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input("Ask a question or continue where we left off..."):
        # Append user message to memory
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Format entire transcript memory for the Gemini client
        formatted_contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part.from_text(text=m["content"])]
            )
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=formatted_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=build_system_instruction(),
                            temperature=0.7,
                        )
                    )
                    reply_text = response.text
                    st.markdown(reply_text)
                    # Append assistant response to memory
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                except Exception as e:
                    st.error(f"Error communicating with Gemini: {e}")

# ------------------------------------------------------------------------------
# TAB 2: Quiz Generator (Uses Learner Memory)
# ------------------------------------------------------------------------------
with tab_quiz:
    st.title("⚡ AI Quiz Generator")

    col1, col2 = st.columns([3, 1])
    with col1:
        quiz_topic = st.text_input("Quiz Topic", value=f"Key concepts in {subject}")
    with col2:
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=10, value=3)

    if st.button("🚀 Generate Quiz"):
        st.session_state.quiz_submitted = False
        with st.spinner("Generating personalized quiz..."):
            mem = st.session_state.user_memory
            quiz_prompt = (
                f"Generate a {num_questions}-question multiple-choice quiz on: {quiz_topic}. "
                f"Target difficulty level: {mem['skill_level']} for learner {mem['name']}."
            )
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=quiz_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=Quiz,
                        temperature=0.4,
                    )
                )
                st.session_state.quiz_data = Quiz.model_validate_json(response.text)
            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")

    # Render Quiz
    if st.session_state.quiz_data:
        quiz = st.session_state.quiz_data
        st.subheader(quiz.title)

        with st.form("quiz_form"):
            user_answers = {}
            for q in quiz.questions:
                st.markdown(f"**Q{q.id}: {q.question}**")
                user_answers[q.id] = st.radio(
                    f"Select answer for Q{q.id}:",
                    options=q.options,
                    key=f"q_{q.id}",
                    label_visibility="collapsed"
                )
                st.markdown("---")

            submitted = st.form_submit_button("Submit Answers")
            if submitted:
                st.session_state.quiz_submitted = True

        if st.session_state.quiz_submitted:
            score = 0
            st.subheader("📊 Quiz Results")

            for q in quiz.questions:
                selected_option = user_answers.get(q.id)
                correct_option = q.options[q.correct_option_index]

                if selected_option == correct_option:
                    score += 1
                    st.success(f"**Q{q.id}: Correct!** ({selected_option})")
                else:
                    st.error(f"**Q{q.id}: Incorrect.** You chose: *{selected_option}* | Correct: *{correct_option}*")

                st.info(f"💡 Explanation: {q.explanation}")

            total = len(quiz.questions)
            st.metric(label="Final Score", value=f"{score} / {total}", delta=f"{int((score / total) * 100)}%")