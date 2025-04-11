const $ = (s) => document.querySelector(s);
const chatBody = $(".chat-body"),
  messageInput = $(".message-input"),
  sendMessage = $("#send-message"),
  fileInput = $("#file-input"),
  fileUploadWrapper = $(".file-upload-wrapper"),
  fileCancelButton = $("#file-cancel"),
  chatbotToggler = $("#chatbot-toggler"),
  closeChatbot = $("#close-chatbot");

const API_KEY = "AIzaSyC9-cNFidYQKoVbni5f7gctXBcB9SwQd90",
  API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${API_KEY}`;

const userData = { message: null, file: {} }, chatHistory = [], initialInputHeight = messageInput.scrollHeight;

const createMessageElement = (content, ...cls) => Object.assign(document.createElement("div"), { className: `message ${cls.join(" ")}`, innerHTML: content });

const generateBotResponse = async (msgDiv) => {
  const msgElem = msgDiv.querySelector(".message-text");
  chatHistory.push({ role: "user", parts: [{ text: userData.message }, ...(userData.file.data ? [{ inline_data: userData.file }] : [])] });
  try {
    const res = await fetch(API_URL, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ contents: chatHistory }) }),
      data = await res.json();
    if (!res.ok) throw new Error(data.error.message);
    const text = data.candidates[0].content.parts[0].text.replace(/\*\*(.*?)\*\*/g, "$1").trim();
    msgElem.innerText = text;
    chatHistory.push({ role: "model", parts: [{ text }] });
  } catch (err) {
    msgElem.innerText = err.message;
    msgElem.style.color = "#f00";
  } finally {
    userData.file = {};
    msgDiv.classList.remove("thinking");
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
  }
};

const handleOutgoingMessage = (e) => {
  e.preventDefault();
  userData.message = messageInput.value.trim();
  if (!userData.message) return;
  messageInput.value = "";
  messageInput.dispatchEvent(new Event("input"));
  fileUploadWrapper.classList.remove("file-uploaded");

  const fileImg = userData.file.data ? `<img src="data:${userData.file.mime_type};base64,${userData.file.data}" class="attachment" />` : "";
  const userMsg = createMessageElement(`<div class="message-text"></div>${fileImg}`, "user-message");
  userMsg.querySelector(".message-text").innerText = userData.message;
  chatBody.appendChild(userMsg);
  chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });

  setTimeout(() => {
    const botMsg = createMessageElement(`
      <img class="bot-avatar" src="screen-0.jpg" alt="Chatbot Logo" width="50" height="50">
      <div class="message-text">
        <div class="thinking-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
      </div>`, "bot-message", "thinking");
    chatBody.appendChild(botMsg);
    chatBody.scrollTo({ top: chatBody.scrollHeight, behavior: "smooth" });
    generateBotResponse(botMsg);
  }, 600);
};

messageInput.addEventListener("input", () => {
  messageInput.style.height = `${initialInputHeight}px`;
  messageInput.style.height = `${messageInput.scrollHeight}px`;
  $(".chat-form").style.borderRadius = messageInput.scrollHeight > initialInputHeight ? "15px" : "32px";
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && messageInput.value.trim() && window.innerWidth > 768) handleOutgoingMessage(e);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    fileInput.value = "";
    fileUploadWrapper.querySelector("img").src = e.target.result;
    fileUploadWrapper.classList.add("file-uploaded");
    userData.file = { data: e.target.result.split(",")[1], mime_type: file.type };
  };
  reader.readAsDataURL(file);
});

fileCancelButton.addEventListener("click", () => (userData.file = {}, fileUploadWrapper.classList.remove("file-uploaded")));

const picker = new EmojiMart.Picker({
  theme: "light", skinTonePosition: "none", previewPosition: "none",
  onEmojiSelect: (emoji) => {
    const { selectionStart, selectionEnd } = messageInput;
    messageInput.setRangeText(emoji.native, selectionStart, selectionEnd, "end");
    messageInput.focus();
  },
  onClickOutside: (e) => document.body.classList.toggle("show-emoji-picker", e.target.id === "emoji-picker")
});

$(".chat-form").appendChild(picker);
sendMessage.addEventListener("click", handleOutgoingMessage);
$("#file-upload").addEventListener("click", () => fileInput.click());
closeChatbot.addEventListener("click", () => document.body.classList.remove("show-chatbot"));
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));
