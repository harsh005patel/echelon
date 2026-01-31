import React, { useState, useEffect, useRef } from 'react';
import { Activity, Radio, Mic, MicOff, AlertTriangle, ShieldCheck, ShieldAlert } from 'lucide-react';
import AudioVisualizer from './AudioVisualizer';
import DetectionStatus from './DetectionStatus';
import AlertPanel from './AlertPanel';
import { connectWebSocket, disconnectWebSocket } from '../services/webrtc';

export default function Dashboard() {
    const [isRecording, setIsRecording] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    // Detection state
    const [currentResult, setCurrentResult] = useState(null);
    const [history, setHistory] = useState([]);

    // Audio Refs
    const audioContextRef = useRef(null);
    const streamRef = useRef(null);
    const processorRef = useRef(null);
    const analyserRef = useRef(null);

    const toggleRecording = async () => {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            // Initialize AudioContext
            audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000, // Match backend
            });

            const source = audioContextRef.current.createMediaStreamSource(stream);

            // Create analyser for visualizer
            const analyser = audioContextRef.current.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser); // source -> analyser
            analyserRef.current = analyser;

            // Create processor to extract raw data
            // Buffer size 4096 is standard for good latency vs performance
            processorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);

            // Connect WS
            const ws = connectWebSocket((data) => {
                handleServerMessage(data);
            });

            ws.onopen = () => setIsConnected(true);
            ws.onclose = () => setIsConnected(false);

            processorRef.current.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                // Convert float32 to int16 for backend efficiency if needed, or send float32
                // Backend expects bytes. Let's send raw float32 bytes or convert to int16.
                // Sending int16 is more bandwidth efficient.
                const pcmData = convertFloat32ToInt16(inputData);

                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(pcmData);
                }
            };

            // source -> analyser -> processor -> destination
            // This ensures both get data. Processor needs destination connection to fire events in some browsers.
            analyser.connect(processorRef.current);
            processorRef.current.connect(audioContextRef.current.destination);

            setIsRecording(true);

        } catch (err) {
            console.error("Error starting recording:", err);
            alert("Could not access microphone");
        }
    };

    const stopRecording = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
        }
        if (processorRef.current) {
            processorRef.current.disconnect();
        }
        disconnectWebSocket();
        setIsRecording(false);
        setIsConnected(false);
        setCurrentResult(null);
        // Analyser will be GC'd with context, or can store ref to disconnect if needed.
    };

    const handleServerMessage = (data) => {
        if (data.is_deepfake !== undefined) {
            setCurrentResult(data);
            setHistory(prev => [data, ...prev].slice(0, 50)); // Keep last 50
        }
    };

    // Helper
    const convertFloat32ToInt16 = (buffer) => {
        let l = buffer.length;
        let buf = new Int16Array(l);
        while (l--) {
            // Clamp
            let s = Math.max(-1, Math.min(1, buffer[l]));
            buf[l] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return buf.buffer;
    };

    return (
        <div className="container">
            {/* Header */}
            <header className="header">
                <div>
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                        Deepfake Voice Guard
                    </h1>
                    <p className="text-muted text-sm mt-1">Real-time VoIP Phishing & Deepfake Detection</p>
                </div>

                <div className="flex items-center gap-4">
                    <div className={`status-badge ${isConnected ? 'status-live' : 'status-offline'}`}>
                        <Activity size={16} />
                        {isConnected ? 'LIVE MONITORING' : 'OFFLINE'}
                    </div>

                    <button
                        onClick={toggleRecording}
                        className={`btn ${isRecording ? 'btn-danger' : 'btn-primary'} flex items-center gap-2`}
                    >
                        {isRecording ? <><MicOff size={18} /> Stop Analysis</> : <><Mic size={18} /> Start Analysis</>}
                    </button>
                </div>
            </header>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* Left Col: Visualizer & Status */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="card h-64 flex flex-col">
                        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Radio size={20} className="text-primary" /> Audio Spectrum
                        </h2>
                        <div className="flex-1 bg-black/20 rounded-lg overflow-hidden relative">
                            <AudioVisualizer isRecording={isRecording} analyser={analyserRef.current} />
                        </div>
                    </div>

                    <DetectionStatus result={currentResult} />
                </div>

                {/* Right Col: Alerts & History */}
                <div className="lg:col-span-1">
                    <AlertPanel history={history} />
                </div>

            </div>
        </div>
    );
}
