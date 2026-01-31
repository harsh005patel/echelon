import React from 'react';
import { ShieldAlert, ShieldCheck, Activity } from 'lucide-react';

export default function DetectionStatus({ result }) {
    if (!result) {
        return (
            <div className="card h-48 flex flex-col justify-center items-center text-center text-muted">
                <Activity size={48} className="mb-4 opacity-20" />
                <p>Waiting for audio stream...</p>
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
                            {isDeepfake ? "POTENTIAL DEEPFAKE DETECTED" : "AUTHENTIC AUDIO DETECTED"}
                        </span>
                    </h2>
                    <p className="text-muted text-sm">
                        Analysis based on Spectral, Prosodic, and Phase features.
                    </p>
                </div>

                <div className={`p-3 rounded-full ${isDeepfake ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
                    {isDeepfake ? <ShieldAlert size={32} /> : <ShieldCheck size={32} />}
                </div>
            </div>

            {/* Confidence Meter */}
            <div className="mt-8">
                <div className="flex justify-between mb-2">
                    <span className="text-sm font-medium">Confidence Score</span>
                    <span className="text-sm font-bold">{confidence.toFixed(1)}%</span>
                </div>
                <div className="h-4 bg-gray-700 rounded-full overflow-hidden relative">
                    <div
                        className={`h-full transition-all duration-300 ${isDeepfake ? 'bg-gradient-to-r from-orange-500 to-red-600' : 'bg-gradient-to-r from-teal-400 to-green-500'}`}
                        style={{ width: `${confidence}%` }}
                    />
                </div>
                <div className="flex justify-between mt-1 text-xs text-muted">
                    <span>Uncertain</span>
                    <span>Highly Confident</span>
                </div>
            </div>

            {/* Detailed Breakdown */}
            {result.individual_scores && (
                <div className="mt-6 pt-4 border-t border-white/5 grid grid-cols-3 gap-4 text-center">
                    <div>
                        <div className="text-xs text-muted mb-1">CNN-LSTM</div>
                        <div className="font-mono text-sm">{(result.individual_scores.cnn_lstm * 100).toFixed(0)}%</div>
                    </div>
                    <div>
                        <div className="text-xs text-muted mb-1">ResNet-SE</div>
                        <div className="font-mono text-sm">{(result.individual_scores.resnet_se * 100).toFixed(0)}%</div>
                    </div>
                    <div>
                        <div className="text-xs text-muted mb-1">RawNet</div>
                        <div className="font-mono text-sm">{(result.individual_scores.rawnet * 100).toFixed(0)}%</div>
                    </div>
                </div>
            )}
        </div>
    );
}
