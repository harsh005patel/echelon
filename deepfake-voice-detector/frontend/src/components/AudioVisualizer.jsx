import React, { useEffect, useRef } from 'react';

export default function AudioVisualizer({ isRecording, analyser }) {
    const canvasRef = useRef(null);
    const animationRef = useRef(null);

    useEffect(() => {
        if (!isRecording || !analyser) {
            if (animationRef.current) cancelAnimationFrame(animationRef.current);
            // clear canvas
            const canvas = canvasRef.current;
            if (canvas) {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Draw idle line
                ctx.strokeStyle = '#334155';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(0, canvas.height / 2);
                ctx.lineTo(canvas.width, canvas.height / 2);
                ctx.stroke();
            }
            return;
        }

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Resize canvas
        const resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                canvas.width = entry.contentRect.width;
                canvas.height = entry.contentRect.height;
            }
        });
        resizeObserver.observe(canvas.parentElement);

        const draw = () => {
            animationRef.current = requestAnimationFrame(draw);

            analyser.getByteFrequencyData(dataArray);

            ctx.fillStyle = '#181b21'; // Should match card bg
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i];

                // Gradient color
                const gradient = ctx.createLinearGradient(0, canvas.height, 0, 0);
                gradient.addColorStop(0, '#6366f1'); // Primary
                gradient.addColorStop(0.5, '#ec4899'); // Accent
                gradient.addColorStop(1, '#a855f7');

                ctx.fillStyle = gradient;

                // Rounded bars logic a bit complex for simple canvas, standard rects for now
                ctx.fillRect(x, canvas.height - barHeight / 1.5, barWidth, barHeight / 1.5);

                x += barWidth + 1;
            }
        };

        draw();

        return () => {
            cancelAnimationFrame(animationRef.current);
            resizeObserver.disconnect();
        };
    }, [isRecording, analyser]);

    return (
        <canvas
            ref={canvasRef}
            className="w-full h-full block"
            width={800}
            height={200}
        />
    );
}
