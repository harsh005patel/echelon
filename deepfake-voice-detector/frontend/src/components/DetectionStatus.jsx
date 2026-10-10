import React from 'react';
import { ShieldAlert, ShieldCheck, Activity, Loader2, Mic, Volume2 } from 'lucide-react';

export default function DetectionStatus({ result, isRecording, isUploading }) {
    if (isUploading) {
        return (
            <div className="card h-48 flex flex-col justify-center items-center text-center text-muted">
                <Loader2 size={40} className="mb-3 animate-spin text-primary" />
                <p className="text-white font-medium">Analyzing audio with ensemble models...</p>
                <p className="text-xs text-muted mt-1">Extracting MFCCs, Mel-Spectrogram & Raw waveform</p>
            </div>
        );
    }

    if (!result) {
        if (isRecording) {
            return (
                <div className="card h-48 flex flex-col justify-center items-center text-center">
                    <div className="relative mb-3">
                        <Mic size={36} className="text-primary animate-pulse" />
                    </div>
                    <p className="text-white font-medium">Microphone stream active</p>
                    <p className="text-xs text-muted mt-1">Speak into microphone — analyzing in 3-second sliding windows</p>
                </div>
            );
        }

        return (
            <div className="card h-48 flex flex-col justify-center items-center text-center text-muted">
                <Activity size={48} className="mb-4 opacity-20" />
                <p className="text-white font-medium">Awaiting Audio Stream</p>
                <p className="text-xs text-muted mt-1">Click "Start Live Mic" or "Upload Audio File" above to begin analysis</p>
            </div>
        );
    }

    // Silence handling
    if (result.status === 'silence') {
        return (
            <div className="card border-blue-500/30 bg-blue-900/10">
                <div className="flex items-start justify-between">
                    <div>
                        <h2 className="text-xl font-bold mb-1 flex items-center gap-2 text-blue-400">
                            <Volume2 size={20} /> Silence / Low Audio Detected
                        </h2>
                        <p className="text-muted text-sm">
                            Audio level is below threshold. Speak clearly into the microphone.
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    const isDeepfake = result.is_deepfake;
    const confidence = result.confidence * 100;

    return (
        <div className={`card transition-colors duration-500 ${isDeepfake ? 'border-red-500/50 bg-red-900/10' : 'border-green-500/50 bg-green-900/10'}`}>
            <div className="flex items-start justify-between">
                <div>
                    <h2 className="text-xl font-bold mb-2 flex items-center gap-2">
                        Status:
                        <span className={isDeepfake ? "text-red-500" : "text-green-500"}>
                            {isDeepfake ? "POTENTIAL DEEPFAKE DETECTED" : "AUTHENTIC VOICE DETECTED"}
                        </span>
                    </h2>
                    <p className="text-muted text-sm">
                        Tri-model ensemble: Spectral MFCC, Mel ResNet-SE & Raw waveform GRU.
                    </p>
                </div>

                <div className={`p-3 rounded-full ${isDeepfake ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
                    {isDeepfake ? <ShieldAlert size={32} /> : <ShieldCheck size={32} />}
                </div>
            </div>

            {/* Confidence Meter */}
            <div className="mt-8">
                <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Ensemble Fake Probability</span>
                    <span className="text-sm font-bold">{confidence.toFixed(1)}%</span>
                </div>
                <div className="h-4 bg-gray-700 rounded-full overflow-hidden relative">
                    <div
                        className={`h-full transition-all duration-300 ${isDeepfake ? 'bg-gradient-to-r from-orange-500 to-red-600' : 'bg-gradient-to-r from-teal-400 to-green-500'}`}
                        style={{ width: `${confidence}%` }}
                    />
                </div>
                <div className="flex justify-between mt-1 text-xs text-muted">
                    <span>Authentic (0%)</span>
                    <span>Synthetic / Deepfake (100%)</span>
                </div>
            </div>

            {/* Detailed Breakdown */}
            {result.individual_scores && (
                <div className="mt-6 pt-4 border-t border-white/5 grid grid-cols-3 gap-4 text-center">
                    <div className="p-2 rounded bg-black/20">
                        <div className="text-xs text-muted mb-1 font-medium">Model 1: CNN-LSTM</div>
                        <div className="font-mono text-sm font-bold text-white mb-1">
                            {(result.individual_scores.cnn_lstm * 100).toFixed(1)}%
                        </div>
                        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-primary"
                                style={{ width: `${result.individual_scores.cnn_lstm * 100}%` }}
                            />
                        </div>
                    </div>
                    <div className="p-2 rounded bg-black/20">
                        <div className="text-xs text-muted mb-1 font-medium">Model 2: ResNet-SE</div>
                        <div className="font-mono text-sm font-bold text-white mb-1">
                            {(result.individual_scores.resnet_se * 100).toFixed(1)}%
                        </div>
                        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-accent"
                                style={{ width: `${result.individual_scores.resnet_se * 100}%` }}
                            />
                        </div>
                    </div>
                    <div className="p-2 rounded bg-black/20">
                        <div className="text-xs text-muted mb-1 font-medium">Model 3: RawNet</div>
                        <div className="font-mono text-sm font-bold text-white mb-1">
                            {(result.individual_scores.rawnet * 100).toFixed(1)}%
                        </div>
                        <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-teal-400"
                                style={{ width: `${result.individual_scores.rawnet * 100}%` }}
                            />
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
