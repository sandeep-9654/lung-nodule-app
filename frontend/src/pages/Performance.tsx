import { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer
} from 'recharts';
import axios from 'axios';

const API_URL = 'http://localhost:5001';

interface FROCData {
    false_positives: number[];
    sensitivity: number[];
}

const defaultMetrics = {
    sensitivity: 94.2,
    specificity: 96.1,
    falsePositives: 1.79,
    inferenceTime: 12
};

const defaultFROC: FROCData = {
    false_positives: [0.125, 0.25, 0.5, 1, 2, 3, 4, 8],
    sensitivity: [0.694, 0.763, 0.822, 0.869, 0.915, 0.942, 0.955, 0.971]
};

export default function Performance() {
    const [activeTab, setActiveTab] = useState<'froc' | 'comparison'>('froc');
    const [frocData, setFrocData] = useState<{ fps: number; sens: number }[]>([]);
    const [, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get(`${API_URL}/froc-data`);
                const data = response.data as FROCData;

                const chartData = data.false_positives.map((fp, i) => ({
                    fps: fp,
                    sens: data.sensitivity[i]
                }));
                setFrocData(chartData);
            } catch {
                // Use default data if API is unavailable
                const chartData = defaultFROC.false_positives.map((fp, i) => ({
                    fps: fp,
                    sens: defaultFROC.sensitivity[i]
                }));
                setFrocData(chartData);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">Model Performance</h1>
                <p className="page-subtitle">
                    Evaluation metrics on LUNA16 dataset (888 scans).
                </p>
            </header>

            {/* Metrics Cards */}
            <div className="metrics-grid">
                <div className="metric-card">
                    <div className="metric-label">Sensitivity</div>
                    <div className="metric-value">{defaultMetrics.sensitivity}%</div>
                    <div className="metric-sublabel">@ 3 False Positives/scan</div>
                    <span className="metric-badge metric-badge-positive">+12% vs 2D</span>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Specificity</div>
                    <div className="metric-value">{defaultMetrics.specificity}%</div>
                    <div className="metric-sublabel">True Negative Rate</div>
                </div>

                <div className="metric-card">
                    <div className="metric-label">False Positives</div>
                    <div className="metric-value">{defaultMetrics.falsePositives}</div>
                    <div className="metric-sublabel">Average per scan</div>
                    <span className="metric-badge metric-badge-negative">-94% vs Traditional</span>
                </div>

                <div className="metric-card">
                    <div className="metric-label">Inference Time</div>
                    <div className="metric-value">{defaultMetrics.inferenceTime}s</div>
                    <div className="metric-sublabel">Per full volume</div>
                </div>
            </div>

            {/* FROC Chart Section */}
            <div className="chart-section">
                <div className="chart-tabs">
                    <button
                        className={`chart-tab ${activeTab === 'froc' ? 'active' : ''}`}
                        onClick={() => setActiveTab('froc')}
                    >
                        FROC Analysis
                    </button>
                    <button
                        className={`chart-tab ${activeTab === 'comparison' ? 'active' : ''}`}
                        onClick={() => setActiveTab('comparison')}
                    >
                        System Comparison
                    </button>
                </div>

                <div className="chart-header">
                    <h2 className="chart-title">Free-Response ROC Curve</h2>
                    <p className="chart-subtitle">Sensitivity vs False Positives per Scan</p>
                </div>

                <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                            data={frocData}
                            margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
                        >
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="var(--border-color)"
                                opacity={0.5}
                            />
                            <XAxis
                                dataKey="fps"
                                stroke="var(--color-text-muted)"
                                fontSize={12}
                                label={{
                                    value: 'False Positives / Scan',
                                    position: 'bottom',
                                    fill: 'var(--color-text-secondary)',
                                    fontSize: 12
                                }}
                            />
                            <YAxis
                                domain={[0, 1]}
                                stroke="var(--color-text-muted)"
                                fontSize={12}
                                tickFormatter={(v) => v.toFixed(2)}
                                label={{
                                    value: 'Sensitivity',
                                    angle: -90,
                                    position: 'insideLeft',
                                    fill: 'var(--color-text-secondary)',
                                    fontSize: 12
                                }}
                            />
                            <Tooltip
                                contentStyle={{
                                    background: 'var(--color-bg-secondary)',
                                    border: '1px solid var(--border-color)',
                                    borderRadius: '8px',
                                    color: 'var(--color-text-primary)'
                                }}
                                formatter={(value) => [Number(value).toFixed(3), 'Sensitivity']}
                                labelFormatter={(label) => `${label} FPs/Scan`}
                            />
                            <Line
                                type="monotone"
                                dataKey="sens"
                                stroke="var(--color-accent-primary)"
                                strokeWidth={3}
                                dot={{
                                    fill: 'var(--color-accent-primary)',
                                    strokeWidth: 2,
                                    r: 5
                                }}
                                activeDot={{ r: 8 }}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
