import React, { useState, useEffect, useRef } from 'react';
import { Activity, Radio, Mic, MicOff, AlertTriangle, ShieldCheck, ShieldAlert, Upload, FileAudio, Loader2 } from 'lucide-react';
import AudioVisualizer from './AudioVisualizer';
import DetectionStatus from './DetectionStatus';
import AlertPanel from './AlertPanel';
import { connectWebSocket, disconnectWebSocket } from '../services/webrtc';

export default function Dashboard() {
    const [isRecording, setIsRecording] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [currentResult, setCurrentResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [uploadError, setUploadError] = useState(null);

    // Audio Refs
    const audioContextRef = useRef(null);
    const streamRef = useRef(null);
    const processorRef = useRef(null);
    const analyserRef = useRef(null);
    const fileInputRef = useRef(null);

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

            // Initialize AudioContext (match backend 16kHz)
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            const audioCtx = new AudioContextClass({ sampleRate: 16000 });
            audioContextRef.current = audioCtx;

            if (audioCtx.state === 'suspended') {
                await audioCtx.resume();
            }

            const source = audioCtx.createMediaStreamSource(stream);

            // Create analyser for visualizer
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            analyserRef.current = analyser;

            // Connect WS with status callbacks
            const ws = connectWebSocket(
                (data) => {
                    handleServerMessage({ ...data, source: 'live' });
                },
                () => setIsConnected(true),
                () => setIsConnected(false)
            );

            // Create processor to extract raw PCM
            const processor = audioCtx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmData = convertFloat32ToInt16(inputData);

                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(pcmData);
                }
            };

            // Connect: source -> analyser -> processor -> silentGain -> destination
            // Using silent GainNode prevents microphone feedback loop into computer speakers
            const silentGain = audioCtx.createGain();
            silentGain.gain.value = 0;

            analyser.connect(processor);
            processor.connect(silentGain);
            silentGain.connect(audioCtx.destination);

            setIsRecording(true);
        } catch (err) {
            console.error("Error starting recording:", err);
            alert("Could not access microphone: " + (err.message || "Permission denied"));
        }
    };

    const stopRecording = () => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (processorRef.current) {
            try { processorRef.current.disconnect(); } catch (_) {}
            processorRef.current = null;
        }
        if (audioContextRef.current) {
            try { audioContextRef.current.close(); } catch (_) {}
            audioContextRef.current = null;
        }
        disconnectWebSocket();
        setIsRecording(false);
        setIsConnected(false);
    };

    const handleServerMessage = (data) => {
        if (data.is_deepfake !== undefined) {
            const formatted = {
                ...data,
                timestamp: new Date().toLocaleTimeString(),
            };
            setCurrentResult(formatted);
            setHistory(prev => [formatted, ...prev].slice(0, 50));
        }
    };

    const handleFileUpload = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        setUploadError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            const formatted = {
                ...data,
                fileName: file.name,
                source: 'file',
                timestamp: new Date().toLocaleTimeString(),
            };

            setCurrentResult(formatted);
            setHistory(prev => [formatted, ...prev].slice(0, 50));
        } catch (err) {
            console.error("Upload analysis error:", err);
            setUploadError(err.message || "Failed to analyze file");
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) fileInputRef.current.value = '';
        }
    };

    const convertFloat32ToInt16 = (buffer) => {
        let l = buffer.length;
        let buf = new Int16Array(l);
        while (l--) {
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
                    <p className="text-muted text-sm mt-1">Real-time VoIP Phishing & Audio Deepfake Detection</p>
                </div>

                <div className="flex items-center gap-4">
                    <div className={`status-badge ${isConnected ? 'status-live' : 'status-offline'}`}>
                        <Activity size={16} />
                        {isConnected ? 'LIVE MONITORING' : 'READY'}
                    </div>

                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        accept="audio/*,.wav,.mp3,.ogg,.flac,.m4a"
                        style={{ display: 'none' }}
                    />

                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="btn bg-gray-700 text-white flex items-center gap-2 hover:bg-gray-600"
                    >
                        {isUploading ? (
                            <><Loader2 size={18} className="animate-spin" /> Analyzing...</>
                        ) : (
                            <><Upload size={18} /> Upload Audio File</>
                        )}
                    </button>

                    <button
                        onClick={toggleRecording}
                        className={`btn ${isRecording ? 'btn-danger' : 'btn-primary'} flex items-center gap-2`}
                    >
                        {isRecording ? <><MicOff size={18} /> Stop Live Mic</> : <><Mic size={18} /> Start Live Mic</>}
                    </button>
                </div>
            </header>

            {uploadError && (
                <div className="p-3 mb-4 rounded-lg bg-red-500/20 border border-red-500/50 text-red-200 text-sm flex items-center gap-2">
                    <AlertTriangle size={16} className="text-red-400" />
                    <span>Upload Error: {uploadError}</span>
                </div>
            )}

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Col: Visualizer & Status */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="card h-64 flex flex-col">
                        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Radio size={20} className="text-primary" /> Audio Spectrum & Signal
                        </h2>
                        <div className="flex-1 bg-black/20 rounded-lg overflow-hidden relative">
                            <AudioVisualizer isRecording={isRecording} analyser={analyserRef.current} />
                        </div>
                    </div>

                    <DetectionStatus result={currentResult} isRecording={isRecording} isUploading={isUploading} />
                </div>

                {/* Right Col: Alerts & History */}
                <div className="lg:col-span-1">
                    <AlertPanel history={history} />
                </div>
            </div>
        </div>
    );
}
