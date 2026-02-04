import { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, Play, Sparkles, AlertCircle } from 'lucide-react';
import axios from 'axios';

const API_URL = 'http://localhost:5001';

interface Nodule {
    id: number;
    centroid: number[];
    size_voxels: number;
    probability: number;
}

interface AnalysisResult {
    success: boolean;
    nodules: Nodule[];
    nodule_count: number;
    inference_time_ms: number;
    volume_shape?: number[];
    slice_data?: number[][];
    prediction_slice?: number[][];
    demo_mode?: boolean;
    message?: string;
}

export default function Analyze() {
    const [file, setFile] = useState<File | null>(null);
    const [isDragOver, setIsDragOver] = useState(false);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [sliceIndex, setSliceIndex] = useState(0);
    const [slices, setSlices] = useState<number[][][]>([]);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Handle file drop
    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile) {
            setFile(droppedFile);
            setError(null);
        }
    }, []);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setIsDragOver(false);
    }, []);

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile) {
            setFile(selectedFile);
            setError(null);
        }
    }, []);

    // Render slice to canvas
    useEffect(() => {
        if (!canvasRef.current || slices.length === 0) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const slice = slices[sliceIndex];
        if (!slice) return;

        const height = slice.length;
        const width = slice[0].length;

        canvas.width = width;
        canvas.height = height;

        const imageData = ctx.createImageData(width, height);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const val = Math.floor(slice[y][x] * 255);
                imageData.data[idx] = val;     // R
                imageData.data[idx + 1] = val; // G
                imageData.data[idx + 2] = val; // B
                imageData.data[idx + 3] = 255; // A
            }
        }

        ctx.putImageData(imageData, 0, 0);
    }, [sliceIndex, slices]);

    // Generate demo slices for visualization
    const generateDemoSlices = () => {
        const numSlices = 64;
        const size = 256;
        const demoSlices: number[][][] = [];

        for (let s = 0; s < numSlices; s++) {
            const slice: number[][] = [];
            for (let y = 0; y < size; y++) {
                const row: number[] = [];
                for (let x = 0; x < size; x++) {
                    // Base noise
                    let val = 0.3 + Math.random() * 0.1;

                    // Left lung
                    const ldist = Math.sqrt((x - 90) ** 2 + (y - 128) ** 2);
                    if (ldist < 60) val = 0.1;

                    // Right lung
                    const rdist = Math.sqrt((x - 166) ** 2 + (y - 128) ** 2);
                    if (rdist < 60) val = 0.1;

                    // Add nodule if in specific slice range
                    if (s >= 40 && s <= 50) {
                        const ndist = Math.sqrt((x - 156) ** 2 + (y - 128) ** 2);
                        if (ndist < 8) val = 0.9;
                    }

                    row.push(val);
                }
                slice.push(row);
            }
            demoSlices.push(slice);
        }

        return demoSlices;
    };

    // Run analysis
    const runAnalysis = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await axios.post<AnalysisResult>(`${API_URL}/predict`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000 // 2 minute timeout
            });

            setResult(response.data);

            // If we got slice data, use it
            if (response.data.slice_data) {
                // For now, wrap single slice in array
                setSlices([response.data.slice_data]);
            } else {
                // Generate demo slices for visualization
                setSlices(generateDemoSlices());
            }

            setSliceIndex(32); // Start at middle

        } catch (err) {
            console.error('Analysis error:', err);
            // Use demo mode if backend is unavailable
            setResult({
                success: true,
                demo_mode: true,
                message: 'Backend not available. Showing demo results.',
                nodules: [
                    { id: 1, centroid: [45, 128, 156], size_voxels: 85, probability: 0.92 },
                    { id: 2, centroid: [32, 245, 198], size_voxels: 62, probability: 0.87 }
                ],
                nodule_count: 2,
                inference_time_ms: 12500
            });
            setSlices(generateDemoSlices());
            setSliceIndex(32);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Analyze CT Scan</h1>
                <p className="page-subtitle">
                    Upload a ZIP file containing the scan pair (.mhd + .raw)
                </p>
            </header>

            <div className="analyze-layout">
                <div>
                    {/* Upload Zone */}
                    <div
                        className={`upload-zone ${isDragOver ? 'drag-over' : ''}`}
                        onDrop={handleDrop}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".zip,.mhd"
                            onChange={handleFileSelect}
                            style={{ display: 'none' }}
                        />

                        <div className="upload-icon">
                            <Upload size={32} />
                        </div>

                        {file ? (
                            <>
                                <p className="upload-text">{file.name}</p>
                                <p className="upload-subtext">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            </>
                        ) : (
                            <>
                                <p className="upload-text">Drag & Drop Scan ZIP</p>
                                <p className="upload-subtext">Must contain both .mhd and .raw files</p>
                            </>
                        )}

                        <button className="btn btn-secondary">Browse Files</button>
                    </div>

                    {/* Slice Viewer */}
                    {slices.length > 0 && (
                        <div className="slice-viewer">
                            <div className="slice-canvas-container">
                                <canvas ref={canvasRef} className="slice-canvas" />
                            </div>
                            <div className="slice-controls">
                                <Play size={16} />
                                <span className="slice-label">SLICE {sliceIndex}</span>
                                <input
                                    type="range"
                                    className="slice-slider"
                                    min={0}
                                    max={slices.length - 1}
                                    value={sliceIndex}
                                    onChange={(e) => setSliceIndex(parseInt(e.target.value))}
                                />
                                <span className="slice-label">{slices.length - 1}</span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Results Panel */}
                <div className="results-panel">
                    <div className="results-header">
                        <Sparkles className="results-icon" size={24} />
                        <span className="results-title">Analysis Results</span>
                    </div>

                    {error && (
                        <div style={{ color: 'var(--color-error)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <AlertCircle size={16} />
                            {error}
                        </div>
                    )}

                    {!result ? (
                        <div className="results-placeholder">
                            <div className="results-placeholder-icon" />
                            <p>Upload a scan and start analysis to view detected nodules.</p>
                        </div>
                    ) : (
                        <div className="nodule-list">
                            {result.demo_mode && (
                                <div style={{
                                    background: 'rgba(243, 156, 18, 0.1)',
                                    border: '1px solid rgba(243, 156, 18, 0.3)',
                                    borderRadius: '8px',
                                    padding: '12px',
                                    marginBottom: '16px',
                                    color: 'var(--color-warning)',
                                    fontSize: '0.875rem'
                                }}>
                                    Demo Mode: Showing simulated results
                                </div>
                            )}

                            <div style={{ marginBottom: '16px', padding: '12px', background: 'var(--color-bg-tertiary)', borderRadius: '8px' }}>
                                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginBottom: '4px' }}>NODULES DETECTED</div>
                                <div style={{ fontSize: '2rem', fontWeight: 700 }}>{result.nodule_count}</div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
                                    in {(result.inference_time_ms / 1000).toFixed(1)}s
                                </div>
                            </div>

                            {result.nodules.map((nodule) => (
                                <div key={nodule.id} className="nodule-item">
                                    <div className="nodule-header">
                                        <span className="nodule-id">Nodule #{nodule.id}</span>
                                        <span className="nodule-confidence">
                                            {(nodule.probability * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="nodule-detail">
                                        Position: ({nodule.centroid.join(', ')})
                                    </div>
                                    <div className="nodule-detail">
                                        Size: {nodule.size_voxels} voxels
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    <button
                        className="btn btn-primary"
                        style={{ width: '100%', marginTop: '1.5rem' }}
                        onClick={runAnalysis}
                        disabled={!file || loading}
                    >
                        {loading ? (
                            <>
                                <div className="loading-spinner" style={{ width: 20, height: 20 }} />
                                Analyzing...
                            </>
                        ) : (
                            'Run AI Diagnosis'
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
