let websocket = null;

export const connectWebSocket = (onMessage, onOpen, onClose) => {
    if (websocket) {
        try {
            websocket.close();
        } catch (_) {}
        websocket = null;
    }

    // In local dev, Vite runs on port 5173 and backend on port 8000
    // Connecting directly to localhost:8000 avoids Vite proxy buffering binary audio frames
    const isDev = window.location.port === '5173' || window.location.hostname === 'localhost';
    const host = isDev ? 'localhost:8000' : window.location.host;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${host}/ws/audio`;

    websocket = new WebSocket(wsUrl);

    websocket.onopen = (event) => {
        if (onOpen) onOpen(event);
    };

    websocket.onclose = (event) => {
        if (onClose) onClose(event);
    };

    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (onMessage) onMessage(data);
        } catch (e) {
            console.error("Error parsing WS message:", e);
        }
    };

    websocket.onerror = (error) => {
        console.error("WebSocket Error:", error);
    };

    return websocket;
};

export const disconnectWebSocket = () => {
    if (websocket) {
        try {
            websocket.close();
        } catch (_) {}
        websocket = null;
    }
};
