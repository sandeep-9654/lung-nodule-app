import { Link } from 'react-router-dom';
import { ArrowRight, Scan, Target, Zap, Heart } from 'lucide-react';

const features = [
    {
        icon: Scan,
        title: '3D Deep Learning',
        description: 'Analyzes full volumetric data using 3D U-Net architectures with residual connections, preserving spatial context across slices.'
    },
    {
        icon: Target,
        title: 'High Sensitivity',
        description: 'Achieves 94.2% sensitivity with only 1.79 false positives per scan, matching state-of-the-art performance.'
    },
    {
        icon: Zap,
        title: 'Instant Triage',
        description: 'Acts as a second reader to clear normal scans quickly and highlight suspicious regions for radiologists.'
    }
];

export default function Home() {
    return (
        <div>
            <section className="hero">
                <div className="hero-badge">
                    <Heart size={16} />
                    <span>Pulmonary Health AI</span>
                </div>

                <h1 className="hero-title">
                    Detect Lung Nodules with<br />
                    <span className="hero-title-accent">Precision AI</span>
                </h1>

                <p className="hero-description">
                    Advanced 3D U-Net Deep Learning architecture for automated localization
                    and classification of pulmonary nodules in volumetric CT scans.
                </p>

                <div className="hero-buttons">
                    <Link to="/analyze" className="btn btn-primary btn-lg">
                        Analyze Scan
                        <ArrowRight size={20} />
                    </Link>
                    <Link to="/performance" className="btn btn-secondary btn-lg">
                        View Metrics
                    </Link>
                </div>
            </section>

            <section className="feature-grid">
                {features.map((feature, index) => (
                    <div key={index} className="feature-card">
                        <div className="feature-icon">
                            <feature.icon size={26} />
                        </div>
                        <h3 className="feature-title">{feature.title}</h3>
                        <p className="feature-description">{feature.description}</p>
                    </div>
                ))}
            </section>
        </div>
    );
}
