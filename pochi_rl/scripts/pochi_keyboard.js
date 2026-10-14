/* Same-origin held-key controls for the Pochi Viser viewer. */
(() => {
  const motion = new Set(["KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"]);
  const pressed = new Set();
  let enabled = true;
  // Keep the control heartbeat separate from the 3D renderer's main thread.
  function controlWorker() {
    let socket, keys = [], sequence = 0, closing = false;
    const send = () => {
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({keys, sequence: sequence++}));
      }
    };
    const connect = url => {
      socket = new WebSocket(url);
      socket.onopen = () => { keys = []; send(); postMessage({type: "open"}); };
      socket.onmessage = event => postMessage({type: "state", state: JSON.parse(event.data)});
      socket.onclose = () => {
        keys = []; postMessage({type: "close"});
        if (!closing) setTimeout(() => connect(url), 1000);
      };
    };
    onmessage = event => {
      const message = event.data;
      if (message.type === "connect") connect(message.url);
      if (message.type === "keys") { keys = message.keys; send(); }
      if (message.type === "close") { closing = true; keys = []; send(); socket?.close(); }
    };
    setInterval(send, 100);
  }
  const workerUrl = URL.createObjectURL(new Blob(["(" + controlWorker.toString() + ")()"], {type: "application/javascript"}));
  const worker = new Worker(workerUrl);
  URL.revokeObjectURL(workerUrl);
  const panel = document.createElement("div");
  panel.id = "pochi-keyboard-panel";
  panel.tabIndex = 0;
  panel.style.cssText = "position:fixed;left:16px;bottom:16px;z-index:9999;background:#17212eee;color:white;padding:12px 16px;border:1px solid #526982;border-radius:10px;font:14px sans-serif;max-width:390px;line-height:1.6";
  panel.innerHTML = '<strong>Pochi V19・6000更新 / WASDQE 操縦</strong><br>W/S：前後　A/D：左右　Q/E：左/右旋回<br>同時押し可・離すと停止・Space/Esc：停止<br><button id="pochi-focus">ここをクリックして操縦</button> <button id="pochi-stop">停止</button><br><span id="pochi-connection">接続中…</span><br><span id="pochi-command"></span><br><span id="pochi-actual"></span>';
  document.body.append(panel);
  const send = () => {
    worker.postMessage({type: "keys", keys: [...pressed].map(k => k.slice(3).toLowerCase())});
  };
  const stop = () => { pressed.clear(); send(); };
  document.getElementById("pochi-focus").onclick = () => panel.focus();
  document.getElementById("pochi-stop").onclick = () => { stop(); panel.focus(); };
  const editing = element => element instanceof Element && Boolean(element.closest("input,textarea,select,[contenteditable=true],[role=textbox],[role=spinbutton],[role=slider]"));
  document.addEventListener("keydown", event => {
    if (event.code === "Escape" || event.code === "Space") {
      if (event.code === "Escape" || !editing(event.target)) {
        stop(); event.preventDefault(); event.stopImmediatePropagation();
      }
      return;
    }
    if (!enabled || event.ctrlKey || event.metaKey || event.altKey || editing(event.target) || !motion.has(event.code)) return;
    event.preventDefault(); event.stopImmediatePropagation();
    if (!event.repeat) { pressed.add(event.code); send(); }
  }, true);
  document.addEventListener("keyup", event => {
    if (pressed.delete(event.code)) { send(); event.preventDefault(); event.stopImmediatePropagation(); }
  }, true);
  document.addEventListener("focusin", event => { if (editing(event.target)) stop(); });
  window.addEventListener("blur", stop);
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  window.addEventListener("pagehide", () => { stop(); worker.postMessage({type: "close"}); });
  const line = (prefix, values) => prefix + " x " + values[0].toFixed(2) + " / y " + values[1].toFixed(2) + " / w " + values[2].toFixed(2);
  worker.onmessage = event => {
    if (event.data.type === "open") { pressed.clear(); send(); }
    if (event.data.type === "state") {
      const state = event.data.state;
      enabled = state.keyboard_enabled;
      if (!enabled && pressed.size) stop();
      document.getElementById("pochi-connection").textContent = state.busy ? "別の画面で操縦中" : !state.ready ? "モデルを準備中…" : enabled ? "キーボード接続済み" : "スライダー操作モード";
      document.getElementById("pochi-command").textContent = line("CMD", state.command ?? [0,0,0]);
      document.getElementById("pochi-actual").textContent = line("ACT", state.actual ?? [0,0,0]);
      panel.dataset.state = JSON.stringify(state);
    }
    if (event.data.type === "close") {
      pressed.clear(); document.getElementById("pochi-connection").textContent = "切断・停止 / 再接続中…";
    }
  };
  worker.postMessage({type: "connect", url: (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/pochi-control"});
})();
