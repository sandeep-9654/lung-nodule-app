import { NavLink } from 'react-router-dom';
import {
    LayoutGrid,
    FileSearch,
    BarChart3,
    Info,
    Scan
} from 'lucide-react';

const navItems = [
    { to: '/', icon: LayoutGrid, label: 'Overview' },
    { to: '/analyze', icon: FileSearch, label: 'Analyze' },
    { to: '/performance', icon: BarChart3, label: 'Performance' },
    { to: '/about', icon: Info, label: 'Project Info' },
];

export default function Sidebar() {
    return (
        <aside className="sidebar">
            <div className="sidebar-brand">
                <div className="sidebar-logo">
                    <Scan size={24} />
                </div>
                <div className="sidebar-brand-text">
                    <span className="sidebar-brand-name">LungCAD AI</span>
                    <span className="sidebar-brand-subtitle">Volumetric Analysis</span>
                </div>
            </div>

            <nav className="sidebar-nav">
                {navItems.map((item) => (
                    <NavLink
                        key={item.to}
                        to={item.to}
                        className={({ isActive }) =>
                            `sidebar-link ${isActive ? 'active' : ''}`
                        }
                    >
                        <item.icon size={20} />
                        <span>{item.label}</span>
                    </NavLink>
                ))}
            </nav>

            <div className="sidebar-status">
                <div className="sidebar-status-indicator">
                    <span className="status-dot" />
                    <span className="sidebar-status-text">System Online</span>
                </div>
                <div className="sidebar-status-version">v2.0.0 (Stable)</div>
                <div className="sidebar-status-version">AI: Active</div>
            </div>
        </aside>
    );
}
