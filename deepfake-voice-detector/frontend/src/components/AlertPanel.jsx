import React from 'react';
import { AlertTriangle, Info, CheckCircle2 } from 'lucide-react';

export default function AlertPanel({ history }) {
    return (
        <div className="card h-full max-h-[600px] flex flex-col">
            <h2 className="text-lg font-semibold mb-4 flex items-center justify-between">
                <span>Detection Log</span>
                <span className="text-xs bg-gray-700 px-2 py-1 rounded text-white">{history.length}</span>
            </h2>

            <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
                {history.length === 0 ? (
                    <div className="text-muted text-center py-10 text-sm">
                        No events detected yet.
                    </div>
                ) : (
                    history.map((event, i) => (
                        <div
                            key={i}
                            className={`p-3 rounded-lg border text-sm animate-in fade-in slide-in-from-right-4 duration-300 ${event.is_deepfake
                                    ? 'bg-red-500/5 border-red-500/20'
                                    : 'bg-green-500/5 border-green-500/20'
                                }`}
                        >
                            <div className="flex items-start gap-3">
                                <div className={`mt-0.5 ${event.is_deepfake ? 'text-red-400' : 'text-green-400'}`}>
                                    {event.is_deepfake ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex justify-between items-start mb-1">
                                        <span className={`font-medium ${event.is_deepfake ? 'text-red-200' : 'text-green-200'}`}>
                                            {event.is_deepfake ? 'Deepfake Detected' : 'Authentic Audio'}
                                        </span>
                                        <span className="text-xs text-muted whitespace-nowrap ml-2">
                                            {/* Assuming we might have timestamps later, for now just 'Now' or index if fast */}
                                            #{i + 1}
                                        </span>
                                    </div>
                                    <div className="flex justify-between text-xs text-muted">
                                        <span>Conf: {(event.confidence * 100).toFixed(1)}%</span>
                                        <span>{event.process_time ? `${(event.process_time * 1000).toFixed(0)}ms` : ''}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
