(() => {
  const messages = document.getElementById("rag-chat-messages");
  const form = document.getElementById("rag-chat-form");
  const input = document.getElementById("rag-chat-input");
  const sendButton = document.getElementById("rag-chat-send");
  const newButton = document.getElementById("rag-chat-new");
  const status = document.getElementById("rag-chat-status");
  const welcome = document.getElementById("chat-welcome");
  if (!messages || !form || !input || !sendButton || !newButton || !status) return;

  let pending = false;

  function appendText(parent, text) {
    const pieces = String(text).split(/\*\*(.+?)\*\*/g);
    pieces.forEach((piece, index) => {
      if (index % 2) {
        const strong = document.createElement("strong");
        strong.textContent = piece;
        parent.appendChild(strong);
      } else {
        parent.appendChild(document.createTextNode(piece));
      }
    });
  }

  function renderMessage(role, text, sources = []) {
    if (welcome?.isConnected) welcome.remove();
    const row = document.createElement("article");
    row.className = `chat-message ${role === "user" ? "from-user" : "from-assistant"}`;
    row.setAttribute("aria-label", role === "user" ? "You" : "IntelliCrew");
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble";

    let list = null;
    String(text || "").split("\n").forEach((line) => {
      const match = line.match(/^\s*[-*]\s+(.+)$/);
      if (match) {
        if (!list) {
          list = document.createElement("ul");
          bubble.appendChild(list);
        }
        const item = document.createElement("li");
        appendText(item, match[1]);
        list.appendChild(item);
        return;
      }
      list = null;
      const paragraph = document.createElement("p");
      appendText(paragraph, line);
      bubble.appendChild(paragraph);
    });
    row.appendChild(bubble);

    if (role === "assistant" && Array.isArray(sources) && sources.length) {
      const sourceList = document.createElement("div");
      sourceList.className = "chat-sources";
      sources.forEach((source) => {
        const chip = document.createElement("span");
        chip.className = "chat-source";
        chip.textContent = [source.employee_id, source.section].filter(Boolean).join(" · ")
          || source.source || source.type || "Source";
        sourceList.appendChild(chip);
      });
      row.appendChild(sourceList);
    }

    messages.appendChild(row);
    messages.scrollTop = messages.scrollHeight;
  }

  function setPending(value) {
    pending = value;
    input.disabled = value;
    sendButton.disabled = value;
    newButton.disabled = value;
    if (value) status.textContent = "IntelliCrew is thinking…";
  }

  async function responseData(response) {
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || data.detail || "Chat is unavailable right now.");
    }
    return data;
  }

  async function sendMessage(text = input.value) {
    if (pending || !text.trim()) return;
    status.textContent = "";
    renderMessage("user", text);
    input.value = "";
    input.style.height = "";
    setPending(true);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await responseData(response);
      status.textContent = "";
      renderMessage("assistant", data.answer, data.sources);
    } catch (error) {
      status.textContent = error.message || "Chat is unavailable right now.";
    } finally {
      setPending(false);
      input.focus();
    }
  }

  async function startNewChat() {
    if (pending) return;
    newButton.disabled = true;
    status.textContent = "Starting a new chat…";
    try {
      await responseData(await fetch("/api/chat/clear", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      }));
      messages.replaceChildren();
      if (welcome) messages.appendChild(welcome);
      status.textContent = "";
    } catch (error) {
      status.textContent = error.message || "Could not start a new chat.";
    } finally {
      newButton.disabled = false;
      input.focus();
    }
  }

  async function restoreHistory() {
    try {
      const response = await fetch("/api/chat/history", {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      });
      const history = await responseData(response);
      if (!Array.isArray(history)) throw new Error("Chat history could not be loaded.");
      history.forEach((message) => {
        if (["user", "assistant"].includes(message.role)
            && typeof message.content === "string") {
          renderMessage(message.role, message.content);
        }
      });
    } catch (error) {
      status.textContent = error.message || "Chat history could not be loaded.";
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });
  messages.addEventListener("click", (event) => {
    const prompt = event.target.closest("[data-prompt]");
    if (prompt) sendMessage(prompt.dataset.prompt);
  });
  newButton.addEventListener("click", startNewChat);
  document.getElementById("chat-logout")?.addEventListener("click", async () => {
    try {
      const response = await fetch("/api/logout", {
        method: "POST", credentials: "same-origin",
      });
      const data = await response.json().catch(() => ({}));
      window.location.replace(data.redirect_url || "/");
    } catch (error) {
      window.location.replace("/");
    }
  });

  restoreHistory();
})();
