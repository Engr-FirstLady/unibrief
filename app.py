import re
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="UniBrief",
    page_icon="📚",
    layout="centered"
)

st.title("📚 UniBrief")
st.caption("Every campus update, clearly explained.")
st.write(
    "Ask a question about the sample campus announcements below."
)

# Sample data for development only.
# Replace with verified official university notices before launch.
announcements = [
    {
        "title": "Project Registration Deadline",
        "category": "Academic",
        "text": (
            "All final-year students must submit their project "
            "registration forms to their department by 4:00 p.m. "
            "on 12 November 2026."
        ),
        "source": "https://example.edu/project-registration"
    },
    {
        "title": "Library Opening Hours",
        "category": "Library",
        "text": (
            "The university library is open from 8:00 a.m. "
            "to 8:00 p.m. Monday to Friday."
        ),
        "source": "https://example.edu/library-hours"
    },
    {
        "title": "Student ID Card Collection",
        "category": "Administration",
        "text": (
            "Students who have been notified that their ID cards "
            "are ready should collect them from the student affairs office."
        ),
        "source": "https://example.edu/id-cards"
    }
]

STOP_WORDS = {
    "what", "when", "where", "who", "why", "how",
    "the", "is", "are", "was", "were", "do", "does",
    "did", "a", "an", "and", "or", "to", "for", "of",
    "in", "on", "at", "by", "i", "me", "my", "it",
    "will", "can", "could", "would", "please", "tell"
}

def keywords(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        word for word in words
        if len(word) > 2 and word not in STOP_WORDS
    }

def search_announcements(question):
    question_words = keywords(question)
    ranked = []

    for notice in announcements:
        searchable = keywords(
            notice["title"] + " " +
            notice["category"] + " " +
            notice["text"]
        )
        overlap = question_words & searchable

        if overlap:
            ranked.append((len(overlap), notice))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked, question_words

@st.cache_resource
def load_model():
    model_id = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model

question = st.text_input(
    "Your question",
    placeholder="When is the project registration deadline?"
)

if st.button("Find answer", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        ranked, question_words = search_announcements(question)

        if not ranked or not question_words:
            st.warning(
                "I couldn't find a matching announcement. "
                "Please check the university's official channels."
            )
        else:
            score, notice = ranked[0]
            confidence = score / len(question_words)

            if confidence < 0.25:
                st.warning(
                    "I couldn't find a sufficiently relevant notice "
                    "to answer confidently."
                )
            else:
                with st.spinner("Preparing an answer..."):
                    tokenizer, model = load_model()

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are UniBrief. Answer only from the "
                                "supplied announcement. Never invent "
                                "dates, fees, or requirements. If the "
                                "answer is absent, say it is not stated."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Notice: {notice['title']}\n"
                                f"Text: {notice['text']}\n"
                                f"Question: {question}\n"
                                "Answer briefly."
                            )
                        }
                    ]

                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    inputs = tokenizer(
                        prompt, return_tensors="pt"
                    )

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=False
                        )

                    new_tokens = outputs[0][
                        inputs["input_ids"].shape[1]:
                    ]
                    answer = tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True
                    ).strip()

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source announcement")
                st.write("**" + notice["title"] + "**")
                st.write(notice["text"])
                st.caption(
                    "Development demo: the source URL below is a "
                    "placeholder, not a verified university link."
                )
                st.link_button("Open source", notice["source"])

st.divider()
st.subheader("Sample announcements")

for notice in announcements:
    with st.expander(notice["title"]):
        st.write("**Category:**", notice["category"])
        st.write(notice["text"])
