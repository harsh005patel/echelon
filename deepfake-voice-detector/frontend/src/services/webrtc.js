let websocket = null;

export const connectWebSocket = (onMessage) => {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        return websocket;
    }

    // Connect to backend
    // In dev: localhost:8000
    // In prod: window.location.host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = 'localhost:8000'; // Hardcoded for dev, normally use relative or env
    const wsUrl = `${protocol}//${host}/ws/audio`;

    websocket = new WebSocket(wsUrl);

    websocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            onMessage(data);
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
        websocket.close();
        websocket = null;
    }
};
