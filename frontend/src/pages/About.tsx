import { GraduationCap, Building, Heart, Users } from 'lucide-react';

const teamMembers = [
    { name: 'B. Sandeep Raghavendra', initials: 'SR' },
    { name: 'A. Jaswanth Kumar', initials: 'JK' },
    { name: 'G. Vignan', initials: 'GV' },
    { name: 'J. Ganesh', initials: 'JG' },
    { name: 'K. Madhu', initials: 'KM' },
];

export default function About() {
    return (
        <div>
            <header className="page-header">
                <h1 className="page-title">About the Project</h1>
                <p className="page-subtitle">
                    Volumetric CT Scan Analysis for Pulmonary Nodule Detection
                </p>
            </header>

            {/* Project Guide Section */}
            <section className="about-section">
                <h2 className="about-section-title">
                    <GraduationCap size={24} />
                    Project Guide
                </h2>
                <div className="team-grid">
                    <div className="team-card guide">
                        <div className="team-avatar">
                            JR
                        </div>
                        <h3 className="team-name">J. Ravindra Babu</h3>
                        <p className="team-role">Assistant Professor</p>
                        <p className="team-role">Dept. of CSE - Data Science</p>
                    </div>
                </div>
            </section>

            {/* Team Members Section */}
            <section className="about-section">
                <h2 className="about-section-title">
                    <Users size={24} />
                    Team Members
                </h2>
                <div className="team-grid">
                    {teamMembers.map((member, index) => (
                        <div key={index} className="team-card">
                            <div className="team-avatar">{member.initials}</div>
                            <h3 className="team-name">{member.name}</h3>
                            <p className="team-role">Team Member</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Institution Section */}
            <section className="about-section">
                <h2 className="about-section-title">
                    <Building size={24} />
                    Institution
                </h2>
                <div className="institution-info">
                    <h3 className="institution-name">
                        KKR & KSR Institute of Technology and Sciences
                    </h3>
                    <p className="institution-location">KITS, Guntur</p>
                </div>
            </section>

            {/* Technology Stack */}
            <section className="about-section">
                <h2 className="about-section-title">
                    <Heart size={24} />
                    Technology Stack
                </h2>
                <div className="feature-grid">
                    <div className="feature-card">
                        <h3 className="feature-title">3D U-Net / V-Net</h3>
                        <p className="feature-description">
                            Deep learning architecture with residual connections for volumetric
                            medical image segmentation. Trained on 64×64×32 patches using
                            mixed precision (AMP) for efficiency.
                        </p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">Flask + PyTorch</h3>
                        <p className="feature-description">
                            Python backend handling complex 3D math operations and GPU-accelerated
                            inference. Supports MPS (Apple Silicon), CUDA, and CPU devices.
                        </p>
                    </div>
                    <div className="feature-card">
                        <h3 className="feature-title">React + Vite</h3>
                        <p className="feature-description">
                            High-performance frontend built with React and Vite for
                            fast development. Features slice viewer and real-time analysis.
                        </p>
                    </div>
                </div>
            </section>

            {/* Research References */}
            <section className="about-section">
                <h2 className="about-section-title">Research Foundation</h2>
                <div className="feature-card">
                    <h3 className="feature-title">Citations</h3>
                    <ul style={{
                        listStyle: 'disc',
                        paddingLeft: '1.5rem',
                        color: 'var(--color-text-secondary)',
                        lineHeight: 1.8
                    }}>
                        <li>
                            Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
                            Image Segmentation" (MICCAI 2015)
                        </li>
                        <li>
                            Milletari et al. "V-Net: Fully Convolutional Neural Networks for
                            Volumetric Medical Image Segmentation" (3DV 2016)
                        </li>
                        <li>
                            LUNA16 Challenge: Lung Nodule Analysis 2016 - Grand Challenge
                        </li>
                    </ul>
                </div>
            </section>
        </div>
    );
}
