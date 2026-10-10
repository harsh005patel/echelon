import React, { useState, useEffect, useRef } from 'react';
import { Activity, Radio, Mic, MicOff, AlertTriangle, ShieldCheck, ShieldAlert, Upload, FileAudio, Loader2 } from 'lucide-react';
import AudioVisualizer from './AudioVisualizer';
import DetectionStatus from './DetectionStatus';
import AlertPanel from './AlertPanel';
import { connectWebSocket, disconnectWebSocket } from '../services/webrtc';

// Helper to encode Int16 PCM samples into standard 16kHz WAV Blob
const encodeWAV = (samples, sampleRate = 16000) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); // 16-bit
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        view.setInt16(offset, samples[i], true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
};

export default function Dashboard() {
    const [isRecording, setIsRecording] = useState(false);
    const [isConnected, setIsConnected] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [currentResult, setCurrentResult] = useState(null);
    const [history, setHistory] = useState([]);
    const [uploadError, setUploadError] = useState(null);

    // Audio & Stream Refs
    const audioContextRef = useRef(null);
    const streamRef = useRef(null);
    const processorRef = useRef(null);
    const analyserRef = useRef(null);
    const fileInputRef = useRef(null);
    const wsRef = useRef(null);
    const recordedSamplesRef = useRef([]);

    const toggleRecording = async () => {
        if (isRecording) {
            await stopRecording();
        } else {
            await startRecording();
        }
    };

    const startRecording = async () => {
        try {
            recordedSamplesRef.current = [];

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

            // Analyser for spectrum visualizer
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
            wsRef.current = ws;

            // Create processor to extract raw PCM (buffer 4096 = 256ms at 16kHz)
            const processor = audioCtx.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmBuffer = convertFloat32ToInt16(inputData);

                // Accumulate samples for end-of-capture analysis
                const int16Array = new Int16Array(pcmBuffer);
                recordedSamplesRef.current.push(int16Array);

                // Send real-time chunks over WebSocket
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(pcmBuffer);
                }
            };

            // Prevent audio feedback loop into speakers with silent GainNode
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

    const stopRecording = async () => {
        // 1. Send stop action to backend stream handler
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            try {
                wsRef.current.send(JSON.stringify({ action: "stop" }));
            } catch (_) {}
        }

        // 2. Stop tracks & processor
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

        setIsRecording(false);

        // 3. Process complete captured audio buffer immediately
        const chunks = recordedSamplesRef.current;
        recordedSamplesRef.current = [];

        if (chunks && chunks.length > 0) {
            const totalSamples = chunks.reduce((acc, c) => acc + c.length, 0);

            // If at least ~0.3s of audio was captured (4800 samples at 16kHz)
            if (totalSamples >= 4800) {
                setIsUploading(true);
                const merged = new Int16Array(totalSamples);
                let offset = 0;
                for (const chunk of chunks) {
                    merged.set(chunk, offset);
                    offset += chunk.length;
                }

                const wavBlob = encodeWAV(merged, 16000);
                const formData = new FormData();
                formData.append('file', wavBlob, 'live_recording.wav');

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        body: formData,
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.is_deepfake !== undefined) {
                            const formatted = {
                                ...data,
                                fileName: 'Captured Voice Clip',
                                source: 'live',
                                timestamp: new Date().toLocaleTimeString(),
                            };
                            setCurrentResult(formatted);
                            setHistory(prev => [formatted, ...prev].slice(0, 50));
                        }
                    }
                } catch (err) {
                    console.error("Finalizing live capture error:", err);
                } finally {
                    setIsUploading(false);
                }
            }
        }

        disconnectWebSocket();
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
                        disabled={isUploading}
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
