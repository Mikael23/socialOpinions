import gradio as gr

# Simulated in-memory data
user_friends = {
    "anna@example.com": ["John", "Maria", "David"],
    "john@example.com": ["Anna", "Elena", "Lucas"]
}

# Simulated session state
session = {
    "user": None,
    "friends": []
}

# -------------------------------
def login(email):
    if email in user_friends:
        session["user"] = email
        session["friends"] = user_friends[email]
        return f"✅ Logged in as {email}", gr.update(choices=session["friends"], visible=True), gr.update(visible=True)
    else:
        return f"⚠️ Email not found. Proceeding as guest.", gr.update(choices=[], visible=True), gr.update(visible=True)

# -------------------------------
def send_opinion(friend, opinion):
    if not opinion.strip():
        return "⚠️ Opinion cannot be empty."
    # Simulate API call here
    print(f"📤 Sending opinion: '{opinion}' about {friend} from user {session['user']}")
    return f"✅ Opinion submitted for {friend}: \"{opinion}\""

# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Social Feedback App (Demo with Gradio)")
    
    with gr.Row():
        email_input = gr.Textbox(label="Your Email (Optional Login)", placeholder="anna@example.com")
        login_btn = gr.Button("Log In")

    login_msg = gr.Markdown("")
    
    friend_list = gr.Dropdown(label="Choose a Friend", choices=[], visible=False)
    opinion_input = gr.Textbox(label="Your Opinion", lines=3, visible=False, placeholder="e.g. John is very creative and supportive")
    submit_btn = gr.Button("Send Opinion", visible=True)
    submit_msg = gr.Markdown("")

    login_btn.click(login, inputs=[email_input], outputs=[login_msg, friend_list, opinion_input])
    submit_btn.click(send_opinion, inputs=[friend_list, opinion_input], outputs=submit_msg)

demo.launch()
